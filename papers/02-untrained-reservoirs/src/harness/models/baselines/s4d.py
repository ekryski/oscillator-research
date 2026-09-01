"""Minimal diagonal state-space reference, S4D-style (Gu, Gupta & Re 2022)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from harness.models.baselines.pooled import PooledBaseline, register_probe
from harness.utils.constants import WARMUP_FRAMES


class S4DBaseline(PooledBaseline):
    """Minimal diagonal state-space reference, S4D-style (Gu, Gupta & Re 2022,
    "On the Parameterization and Initialization of Diagonal State Space
    Models"): per-channel learnable complex poles a = exp(-exp(log_decay)
    + i*freq) (log-parameterized decay keeps |a| < 1 for any parameter value),
    n_states diagonal states per channel, input/output mixing linears, a real
    C readout per state pair and a D skip, RECURRENT scan (the streaming-honest
    form; the convolutional view is an optimization we don't need at harness
    scale). freq initialized linspace(0, pi) — poles spread across the whole
    discrete band, the S4D-Lin flavor. 1,840 params (-10.2% of 2,048)."""

    def __init__(self, grid: int = 16, hidden: int = 16, n_states: int = 16,
                 n_classes: int = 8, probe_seed: int = 0):
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
        register_probe(self, 4 * hidden, n_classes, probe_seed)

    def _hidden(self, rows: torch.Tensor) -> torch.Tensor:
        u = self.inp(rows)                                   # [B,T,H]
        a = torch.exp(torch.complex(-torch.exp(self.log_decay), self.freq))  # [H,N], |a|<1
        c = torch.complex(self.c_re, self.c_im)
        x = torch.zeros(rows.shape[0], self.hidden, self.n_states,
                        dtype=a.dtype, device=rows.device)
        ys = []
        for u_t in u.unbind(1):                              # recurrent scan
            x = a * x + self.b * u_t[:, :, None].to(a.dtype)
            ys.append((x * c).sum(dim=-1).real + self.d_skip * u_t)
        y = torch.stack(ys, dim=1)                           # [B,T,H]
        return F.relu(self.out(y))[:, WARMUP_FRAMES:]

    def _hidden_tail(self, rows: torch.Tensor, n_settle: int) -> torch.Tensor:
        """S4D is genuinely recurrent, so concatenating a zero tail IS running
        the recurrence n_settle extra zero-input steps from the final state —
        the GRU-style settle, no asymmetry (input biases stay active, exactly
        as in the GRU's zero-input steps)."""
        return super()._hidden_tail(rows, n_settle)

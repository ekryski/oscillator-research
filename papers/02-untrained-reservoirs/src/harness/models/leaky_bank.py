"""The non-oscillating control: a bank of independent leaky integrators.

The oscillator field is a large frozen bank of units with memory, read by a
linear readout. So is this. What it lacks is the thing under test: nothing
here rotates and nothing is coupled. Each unit relaxes toward its own input at
its own speed,

    x <- (1 - a) x + a tanh(g u)

and never sees another unit. If the field cannot beat this, its accuracy comes
from being a bank of temporal filters and not from being oscillators.

It is matched to the field the way the field is built. The same routing: band r
of the front end drives every unit of lattice row r, across all channels and
columns. The same count: two frozen tensors of shape [channels, grid, grid],
2,048 stored parameters at 4 channels, against the field's coupling kernel and
natural frequencies. The input gains are drawn as the field draws its natural
frequencies, N(1, 0.1^2), seed-pinned. The leak rates are not drawn at all: one
fixed log-spaced schedule of time constants, laid out so that every band is
read at every time scale.

Nothing about the schedule is tuned. This is a falsification control, not a
claim that the design is a good one. Because only the gains vary with the seed,
this arm's seed-to-seed spread is small by construction, and should be read
that way.

Two sizes are used, because the field exposes two signals per state
(sin and cos) and a leaky unit exposes one: `channels=4` matches the field's
states and parameters, `channels=8` matches its exposed signals at twice the
parameters.
"""

from __future__ import annotations

import math

import torch
from torch import nn

#: the frame rate of the hop front end
HOP_RATE_HZ = 62.5
#: time constants span one hop frame to a full one-second clip
TAU_MIN_S, TAU_MAX_S = 1.0 / HOP_RATE_HZ, 1.0
#: input gains are drawn as the field draws its natural frequencies
GAIN_MEAN, GAIN_STD = 1.0, 0.1


class LeakyBank(nn.Module):
    """Independent leaky integrators, frozen, with the field's routing and count."""

    def __init__(self, channels: int = 4, grid: int = 16, gain: float = 2.0, seed: int = 0,
                 rate_hz: float = HOP_RATE_HZ, tau_min_s: float = TAU_MIN_S,
                 tau_max_s: float = TAU_MAX_S):
        super().__init__()
        self.channels, self.grid, self.gain, self.rate_hz = channels, grid, gain, rate_hz
        gen = torch.Generator().manual_seed(seed)
        input_gain = torch.randn(channels, grid, grid, generator=gen) * GAIN_STD + GAIN_MEAN
        # one schedule, shared by every row: the units of a row are its
        # channels x columns, and they run from the fastest to the slowest
        per_row = channels * grid
        tau_row = torch.logspace(math.log10(tau_min_s), math.log10(tau_max_s), per_row)
        tau = tau_row.view(channels, 1, grid).expand(channels, grid, grid).contiguous()
        # frozen, but kept as parameters so they are counted the way the field's are
        self.input_gain = nn.Parameter(input_gain, requires_grad=False)
        self.tau_s = nn.Parameter(tau, requires_grad=False)

    @property
    def n_states(self) -> int:
        return self.channels * self.grid * self.grid

    def leak(self) -> torch.Tensor:
        """Per-step leak rate a for each unit, from its time constant and the row rate.

        Stated in seconds and converted here, so the same bank means the same
        physical filters at any row rate.
        """
        return 1.0 - torch.exp(-1.0 / (self.tau_s * self.rate_hz))

    def signals(self, rows: torch.Tensor) -> torch.Tensor:
        """[B, T, G] front-end rows -> [B, T, states]: every unit's state over time."""
        if rows.dim() != 3:
            raise ValueError("the leaky bank has no phase, so it cannot take the quadrature "
                             f"pathway's paired rows; got shape {tuple(rows.shape)}")
        b, t, g = rows.shape
        if g != self.grid:
            raise ValueError(f"{g} bands cannot drive a lattice of {self.grid} rows")
        a = self.leak().reshape(1, -1)
        gain = self.input_gain.reshape(1, -1)
        x = rows.new_zeros(b, self.n_states)
        out = []
        for step in range(t):
            # band r drives every unit of row r, in all channels and columns.
            # Routed one frame at a time: the whole routed input is T times
            # the size of a state, which long inputs cannot afford.
            band = (rows[:, step] * self.gain).view(b, 1, g, 1)
            routed = band.expand(b, self.channels, g, self.grid).reshape(b, -1)
            x = (1.0 - a) * x + a * torch.tanh(gain * routed)
            out.append(x)
        return torch.stack(out, dim=1)

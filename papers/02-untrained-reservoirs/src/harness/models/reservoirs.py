"""Two reservoir baselines from the reservoir-computing literature, and a memoryless control.

The study's own controls (the uncoupled network and the leaky-integrator bank)
have memory but no mixing between units: no unit ever sees another. What the
standard reservoir adds is a random recurrent matrix that mixes every unit's
state into every other's. These arms supply that, read through the same
statistics, projection and ridge as every other arm.

    EchoStateNetwork   the leaky echo state network (Jaeger 2001; Jaeger et al. 2007),
                           x <- (1 - a) x + a tanh(W x + V (g u) + b)
                       W sparse and random, rescaled to a spectral radius rho; V and b
                       uniform on [-1, 1]; a the leaking rate; g the input scaling,
                       which is the run's input gain. With rho = 0 and a = 1 nothing
                       is remembered and the unit is a random nonlinear expansion of
                       the current frame, tanh(V (g u) + b): the memoryless control.
    RandomOscillators  the random oscillators network of Ceni et al. (AISTATS 2024),
                       N damped harmonic oscillators forced through a random layer,
                           z <- z + tau (tanh(W y + V (g u) + b) - gamma y - eps z)
                           y <- y + tau z
                       with gamma (stiffness) and eps (damping) drawn per oscillator
                       from uniform ranges, W uniform on [-2, 2] rescaled to spectral
                       radius rho, V uniform on [0, 1] scaled by the input gain, b
                       uniform on [-1, 1]. The readout sees y and z, the whole state,
                       as it sees the oscillator network's sin and cos.

Every parameter is drawn once from the seed and never updated. Sizes are matched
to the oscillator network's 1,024 states: 1,024 echo state units, or 512 random
oscillators with their velocities. Hyperparameters are selected on the validation
speakers of Protocol V (harness.experiment.protocol), never on the test set.
"""

from __future__ import annotations

import torch
from torch import nn

#: incoming recurrent connections per echo state unit
ESN_FAN_IN = 10


def _spectral_radius(w: torch.Tensor) -> float:
    return torch.linalg.eigvals(w.double()).abs().max().item()


def _rescaled(w: torch.Tensor, rho: float) -> torch.Tensor:
    """W scaled so its spectral radius is rho (zero if rho is 0)."""
    if rho == 0.0:
        return torch.zeros_like(w)
    return w * (rho / _spectral_radius(w))


class EchoStateNetwork(nn.Module):
    """A leaky echo state network, frozen. `units` states, one signal each."""

    def __init__(self, units: int = 1024, inputs: int = 16, spectral_radius: float = 0.9, leak: float = 0.3,
                 gain: float = 1.0, seed: int = 0, fan_in: int = ESN_FAN_IN):
        super().__init__()
        self.units, self.inputs, self.gain = units, inputs, gain
        self.spectral_radius, self.leak_rate = spectral_radius, leak
        gen = torch.Generator().manual_seed(seed)
        dense = torch.randn(units, units, generator=gen)
        keep = torch.rand(units, units, generator=gen) < (fan_in / units)
        w = _rescaled(dense * keep, spectral_radius)
        v = torch.rand(units, inputs, generator=gen) * 2 - 1
        b = torch.rand(units, generator=gen) * 2 - 1
        self.w = nn.Parameter(w, requires_grad=False)
        self.v = nn.Parameter(v, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)

    @property
    def n_states(self) -> int:
        return self.units

    def signals(self, rows: torch.Tensor) -> torch.Tensor:
        """[B, T, G] front-end rows -> [B, T, units]."""
        if rows.dim() != 3:
            raise ValueError(f"an echo state network takes [B, T, G] rows; got shape {tuple(rows.shape)}")
        b, t, g = rows.shape
        if g != self.inputs:
            raise ValueError(f"{g} bands cannot drive a network built for {self.inputs}")
        a = self.leak_rate
        x = rows.new_zeros(b, self.units)
        drive = (rows * self.gain) @ self.v.T + self.b          # [B, T, units], once
        out = []
        for step in range(t):
            x = (1.0 - a) * x + a * torch.tanh(x @ self.w.T + drive[:, step])
            out.append(x)
        return torch.stack(out, dim=1)


class RandomOscillators(nn.Module):
    """The random oscillators network, frozen. `units` oscillators, two states and two signals each."""

    def __init__(self, units: int = 512, inputs: int = 16, tau: float = 0.1, spectral_radius: float = 1.0,
                 gamma: tuple[float, float] = (1.0, 0.5), eps: tuple[float, float] = (0.5, 0.4),
                 gain: float = 1.0, seed: int = 0):
        super().__init__()
        self.units, self.inputs, self.gain, self.tau = units, inputs, gain, tau
        self.spectral_radius, self.gamma_range, self.eps_range = spectral_radius, tuple(gamma), tuple(eps)
        gen = torch.Generator().manual_seed(seed)
        w = _rescaled(torch.rand(units, units, generator=gen) * 4 - 2, spectral_radius)
        v = torch.rand(units, inputs, generator=gen)
        b = torch.rand(units, generator=gen) * 2 - 1
        g_mid, g_rad = gamma
        e_mid, e_rad = eps
        g = g_mid + (torch.rand(units, generator=gen) * 2 - 1) * g_rad
        e = e_mid + (torch.rand(units, generator=gen) * 2 - 1) * e_rad
        for name, value in (("w", w), ("v", v), ("b", b), ("gamma", g), ("eps", e)):
            setattr(self, name, nn.Parameter(value, requires_grad=False))

    @property
    def n_states(self) -> int:
        return 2 * self.units

    def signals(self, rows: torch.Tensor) -> torch.Tensor:
        """[B, T, G] front-end rows -> [B, T, 2 units]: every oscillator's position, then its velocity."""
        if rows.dim() != 3:
            raise ValueError(f"a random oscillators network takes [B, T, G] rows; got shape {tuple(rows.shape)}")
        b, t, g = rows.shape
        if g != self.inputs:
            raise ValueError(f"{g} bands cannot drive a network built for {self.inputs}")
        tau = self.tau
        y = rows.new_zeros(b, self.units)
        z = rows.new_zeros(b, self.units)
        drive = (rows * self.gain) @ self.v.T + self.b
        out = []
        for step in range(t):
            z = z + tau * (torch.tanh(y @ self.w.T + drive[:, step]) - self.gamma * y - self.eps * z)
            y = y + tau * z
            out.append(torch.cat((y, z), dim=1))
        return torch.stack(out, dim=1)

"""The connectivity-disorder control: random graphs instead of a stencil.

Per-channel RANDOM graphs replace the translation-invariant circulant kernel,
with the same Kuramoto pairwise law, the same omega, pinning and drive
semantics, and the same feature layout — ONLY the coupling topology varies.
That is the point: it answers whether the designed local stencil does anything
a generic connectivity of matched strength would not.

Weights are normalized per channel to the spectral clamp by operator 2-norm.
For the non-normal random W the matching quantity is sigma_max, not the
eigenvalue radius, because the clamp's meaning is an amplification cap.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RandGraphCore(nn.Module):
    """Connectivity-disorder core: per-channel RANDOM graphs replace
    the translation-invariant circulant stencil. Same Kuramoto pairwise law
    (torque_i = sum_j W_ij sin(theta_j - theta_i), computed as
    cos*(W@sin) - sin*(W@cos)), same omega/pinning/drive semantics, same
    feature layout as the phase core ([sin | cos] blocks) -- ONLY the
    coupling topology varies. W: graph_k nonzeros per oscillator (out of
    G*G possible), weights ~ N(0,1), spectrally normalized per channel to
    `spectral_clamp` (the ESN echo-state convention, matching the circulant
    clamp's meaning)."""

    def __init__(self, channels: int, grid: int, dt: float, damping: float,
                 spectral_clamp: float, graph_k: int, seed: int, substeps: int = 1):
        super().__init__()
        self.channels, self.grid, self.dt = channels, grid, dt
        self.damping, self.spectral_clamp, self.substeps = damping, spectral_clamp, substeps
        n = grid * grid
        g = torch.Generator().manual_seed(seed)
        w = torch.zeros(channels, n, n)
        for ch in range(channels):
            for i in range(n):
                cols = torch.randperm(n, generator=g)[:graph_k]
                w[ch, i, cols] = torch.randn(graph_k, generator=g)
            # operator-2-norm normalize to the clamp: the circulant clamp is
            # an AMPLIFICATION cap (max |K_hat|); for the non-normal random W
            # the matching quantity is sigma_max, not the eigenvalue radius
            sv = torch.linalg.matrix_norm(w[ch], ord=2).clamp_min(1e-12)
            w[ch] = w[ch] * (spectral_clamp / sv)
        self.register_buffer("W", w)
        self.natural_freqs = nn.Parameter(
            1.0 + 0.1 * torch.randn(channels, grid, grid,
                                    generator=torch.Generator().manual_seed(seed + 1)))
        theta0 = 2 * math.pi * torch.rand(channels, grid, grid,
                                          generator=torch.Generator().manual_seed(seed + 2))
        self.register_buffer("theta0", theta0)

    def forward_scan(self, drives: torch.Tensor, state: torch.Tensor | None = None):
        b, t = drives.shape[0], drives.shape[1]
        c, g = self.channels, self.grid
        n = g * g
        if state is not None:  # settle read: continue from the driven state
            theta = state.reshape(b, c, n).clone()
        else:
            theta = self.theta0[None].expand(b, c, g, g).reshape(b, c, n).clone()
        om = self.natural_freqs.reshape(1, c, n)
        drv = drives.reshape(b, t, c, n)
        out = torch.empty(b, t, 2 * c * n, device=drives.device)
        for step in range(t):
            for _ in range(self.substeps):
                s_, c_ = torch.sin(theta), torch.cos(theta)
                conv_s = torch.einsum("cij,bcj->bci", self.W, s_)
                conv_c = torch.einsum("cij,bcj->bci", self.W, c_)
                torque = c_ * conv_s - s_ * conv_c
                theta = theta + (self.dt / self.substeps) * (
                    om + torque + drv[:, step] - self.damping * torch.sin(theta))
            out[:, step, :c * n] = torch.sin(theta).reshape(b, c * n)
            out[:, step, c * n:] = torch.cos(theta).reshape(b, c * n)
        return out, theta.reshape(b, c, g, g)

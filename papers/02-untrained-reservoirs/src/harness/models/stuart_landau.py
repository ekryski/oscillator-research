"""Stuart-Landau complex-amplitude core.

Each oscillator carries a complex state z = x + iy on the same C-channel
toroidal grid as the phase core; amplitude is a state variable, fixing the
"[sinθ, cosθ] is unit-norm forever and cannot encode absence" readout gap.

Per Euler substep (the recipe — Cartesian to avoid the 1/r polar
singularity, amplitude implicit for unconditional radial stability):

  1. explicit coupling + pinning + drive:
       z <- z + dt * (K ⊛ z − s0·z + λ(1 − z) + i·drive·z)
     (s0 = per-channel kernel sum, making the coupling diffusive; the same
      spectral clamp bounds the operator norm. Drive enters TANGENTIALLY —
      in continuous time i·d·z gives θ' += d with amplitude untouched —
      matching the phase core's additive-on-θ̇ convention so the A/B isolates
      the amplitude mechanism alone. A real-axis drive z += dt·d is a
      *different* physics — a θ-dependent Adler-type coupling — recorded as a
      candidate follow-up arm, since the findings flagged drive-entry
      form as a lever.

      Discretization ordering (intended): the update is sequential — the y
      line reads the already-updated x (semi-implicit / symplectic-Euler
      ordering). For the drive rotation this is an area-preserving map whose
      radius error is oscillatory O((dt·d)²) with no secular drift; the
      substep-start (Jacobi) alternative would inflate the radius by
      √(1+(dt·d)²) every substep — strictly worse. Either residue is erased
      by step 2, and the two orderings differ only at the scheme's existing
      O(dt²) truncation order (verified numerically: arm-B radial
      difference is float32 noise; phase difference scales as dt²). The TS
      port (viz/src/physics/sl.ts) mirrors this ordering exactly for parity.)
  2. implicit amplitude relaxation (r' = αr − βr³ backward-Euler), solved in
     closed form: dtβ·r³ + (1 − dtα)·r − r_old = 0 has one positive real root
     (Cardano; p > 0 branch), applied as a radial rescale.
  3. exact phase rotation: z <- z · e^{i·dt·ω}.

Phase reduction: at |z| ≡ 1 the tangential projections of (1) are EXACTLY the
kuramoto torque Σ K sin(θ_j − θ_i) and the pinning −λ sinθ (the s0 term is
purely radial) — the phase core is this core's equal-amplitude slice, so arm B
(amplitude frozen, renormalized) reproduces kuramoto up to O(dt²) projection
error. Readout is cat(ℑz, ℜz) to match the phase core's [sinθ, cosθ] ordering:
probes, featurization, and instrumentation are shared unchanged.

Harness-scale only for now: blocks=1, tbptt=0, no torch.compile.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _softplus_inv(v: float) -> float:
    return math.log(math.expm1(v))


def _cbrt(t: torch.Tensor) -> torch.Tensor:
    """Signed cube root, differentiable away from 0."""
    return torch.sign(t) * t.abs().clamp_min(1e-30).pow(1.0 / 3.0)


def implicit_amplitude_root(r_old: torch.Tensor, dt_beta: torch.Tensor,
                            one_minus_dt_alpha: torch.Tensor) -> torch.Tensor:
    """Positive real root of dtβ·r³ + (1−dtα)·r − r_old = 0 via Cardano.

    With dtβ > 0 and 1−dtα > 0 (dt·α ≪ 1 always holds here) the depressed cubic
    has p > 0, hence exactly one real root, and it is positive for r_old ≥ 0."""
    p = one_minus_dt_alpha / dt_beta
    q = -r_old / dt_beta
    disc = (q / 2) ** 2 + (p / 3) ** 3  # > 0 on the p > 0 branch
    s = torch.sqrt(disc)
    return _cbrt(-q / 2 + s) + _cbrt(-q / 2 - s)


class SLCore(nn.Module):
    """Stuart-Landau field with the PhaseCore scan interface.

    amplitude_frozen=True is arm B: α/β untrainable and |z| renormalized
    to 1 every substep — isolates the coupling-form change from the amplitude
    mechanism (the ablation the source paper omitted)."""

    def __init__(self, channels: int = 4, grid: int = 16, substeps: int = 1,
                 dt: float = 0.1, damping: float = 0.5, spectral_clamp: float = 0.5,
                 damping_learnable: bool = False,
                 alpha_init: float = 0.05, amplitude_frozen: bool = False,
                 seed: int = 42):
        super().__init__()
        self.channels, self.grid, self.substeps, self.dt = channels, grid, substeps, dt
        self.damping, self.spectral_clamp = damping, spectral_clamp
        # harness slice: promote lambda to a per-channel physics parameter
        # (raw, init at the scalar value; learned spread IS the finding)
        if damping_learnable:
            self.damping_lam = nn.Parameter(torch.full((channels, 1, 1), float(damping)))
        self.amplitude_frozen = amplitude_frozen
        # same init family as PhaseBlock for comparability
        self.kernel = nn.Parameter(torch.randn(channels, grid, grid) * 0.05)
        self.natural_freqs = nn.Parameter(torch.randn(channels, grid, grid) * 0.1 + 1.0)
        # alpha sets the amplitude relaxation RATE (init 0.05 — the SL-GNN
        # criticality prior: slow, near-marginal); alpha/beta sets the attracting
        # RADIUS — init beta = alpha so r* = 1 and the core starts on the phase
        # core's equal-amplitude slice (resolves the spec's tension between
        # "criticality init" and "init-identical": rate and radius are separate
        # dials). Both learnable per channel (arm C).
        self.alpha = nn.Parameter(torch.full((channels, 1, 1), alpha_init))
        self.beta_hat = nn.Parameter(torch.full((channels, 1, 1), _softplus_inv(alpha_init)))
        if amplitude_frozen:
            self.alpha.requires_grad_(False)
            self.beta_hat.requires_grad_(False)
        gen = torch.Generator().manual_seed(seed)
        theta0 = torch.rand(1, channels, grid, grid, generator=gen) * (2 * math.pi)
        # start on the unit circle at the same seeded angles as the phase core
        self.register_buffer("state0", torch.stack((torch.cos(theta0), torch.sin(theta0)), dim=1))

    @property
    def readout_dim(self) -> int:
        return 2 * self.channels * self.grid * self.grid

    def init_state(self, batch: int) -> torch.Tensor:
        return self.state0.expand(batch, -1, -1, -1, -1).clone()  # [B, 2(x,y), C, G, G]

    def prepare_coupling(self) -> tuple[torch.Tensor, torch.Tensor]:
        kfft = torch.fft.rfft2(self.kernel)
        if self.spectral_clamp:
            mag = kfft.abs().amax(dim=(-2, -1), keepdim=True)
            kfft = kfft * torch.clamp(self.spectral_clamp / (mag + 1e-8), max=1.0)
        s0 = torch.fft.irfft2(kfft, s=(self.grid, self.grid)).sum(dim=(-2, -1))  # [C]
        return kfft, s0.view(1, -1, 1, 1)

    def step_frame(self, state: torch.Tensor, drive: torch.Tensor,
                   coup: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        kfft, s0 = coup
        x, y = state[:, 0], state[:, 1]
        dt = self.dt
        lam = self.damping_lam if hasattr(self, "damping_lam") else self.damping
        beta = torch.nn.functional.softplus(self.beta_hat)
        co, si = torch.cos(dt * self.natural_freqs), torch.sin(dt * self.natural_freqs)
        for _ in range(self.substeps):
            field = torch.stack((x, y), dim=1)
            conv = torch.fft.irfft2(torch.fft.rfft2(field) * kfft, s=(self.grid, self.grid))
            # tangential drive: i·d·z = (−d·y, +d·x) — pure phase push, θ' += d.
            # y reads the updated x on purpose (semi-implicit ordering — see
            # "Discretization ordering" in the module docstring).
            x = x + dt * (conv[:, 0] - s0 * x + lam * (1 - x) - drive * y)
            y = y + dt * (conv[:, 1] - s0 * y - lam * y + drive * x)
            r_old = torch.sqrt(x * x + y * y).clamp_min(1e-8)
            if self.amplitude_frozen:
                scale = 1.0 / r_old
            else:
                r_new = implicit_amplitude_root(r_old, dt * beta, 1 - dt * self.alpha)
                scale = r_new / r_old
            x, y = x * scale, y * scale
            x, y = x * co - y * si, x * si + y * co  # z <- z * e^{i dt omega}
        new_state = torch.stack((x, y), dim=1)
        feat = torch.cat((y, x), dim=1).flatten(1)  # [Im z, Re z] = [sinθ, cosθ] order
        return new_state, feat

    def forward_scan(self, drives: torch.Tensor, state: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """drives [B, T, C, G, G] -> (features [B, T, 2CGG], final state)."""
        b, t = drives.shape[:2]
        if state is None:
            state = self.init_state(b)
        coup = self.prepare_coupling()
        outs = []
        for drive_t in drives.unbind(1):
            state, feat = self.step_frame(state, drive_t, coup)
            outs.append(feat)
        return torch.stack(outs, dim=1), state

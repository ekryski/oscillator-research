"""The phase-oscillator core: C channels of G x G coupled phases on a lattice.

Coupling is translation-invariant, one kernel of weights applied at every
site, so it is a convolution, computed by FFT or as one dense circulant
matmul, whichever the device prefers. The lattice's edge gluing lives in
`harness.models.geometries`; this module is geometry-agnostic.

Per Euler substep (coupling="kuramoto"):

    torque_i = sum_j K(i - j) * sin(theta_j - theta_i)
             = cos(theta_i) * (K * sin theta)_i  -  sin(theta_i) * (K * cos theta)_i

    theta <- theta + dt * (omega + torque - lambda * sin(theta) + drive)

The other coupling functions, at the canonical values paper 02 froze them at:

    sakaguchi   sum_j K sin(theta_j - theta_i - alpha), alpha = pi/4
    harmonic2   Kuramoto + beta * sum_j K sin(2(theta_j - theta_i)), beta = 0.5
    winfree     -sin(theta_i) * sum_j K (1 + cos theta_j)

The quadrature pathway adds a phase-referenced drive (qcos, qsin) =
gain * A * (cos phi, sin phi) per site, whose torque is computed from the
current phase at each substep: gain*A*sin(phi - theta) = qsin*cos(theta) -
qcos*sin(theta). Without it every path is exactly the additive-only step.

The coupling ceiling. Each channel's kernel is scaled so that the peak of its
spectrum, the largest factor by which the coupling amplifies any pattern of
phases across the channel, is exactly the ceiling (`kernel_scaling="exact"`,
paper 03). Paper 02 only ever scaled a kernel down to the ceiling
(`kernel_scaling="cap"`), so a random kernel whose peak was already below it
was left weaker; that happens only on small lattices (about 10% of 8 x 8
channels at a ceiling of 1, never at 16 x 16 or above). Wherever the ceiling
binds the two are the same arithmetic, bit for bit, which is what lets paper
02's 16 x 16 runs stand as paper 03's. For the periodic geometries the
spectral peak is the operator's exact 2-norm; for the padded and weighted ones
(pad, crop and the sphere's weights all have norm at most 1) it is a bound.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from harness.models.geometries import BOUNDARIES, build_geometry

TWO_PI = 2 * math.pi
COUPLINGS = ("kuramoto", "sakaguchi", "harmonic2", "winfree")
#: how a channel's kernel meets the coupling ceiling
KERNEL_SCALINGS = ("exact", "cap")
#: guards the division when a kernel's peak is zero
PEAK_EPS = 1e-8
#: the largest lattice each device couples with the dense operator under "auto"; above it, FFT.
#: Measured per channel and clip: on one CPU thread the dense path is paper 02's at 16 x 16, and FFT
#: is 3.6 times faster at 32 x 32; on MPS (M1 Max) the dense path is 2 to 80 times faster than FFT
#: up to 64 x 64, where its operator is 4,096 x 4,096, and its operator too large at 128 x 128;
#: CUDA uses FFT, as paper 02 did.
DENSE_UP_TO = {"cpu": 16, "mps": 64, "cuda": 0}


def auto_impl(device_type: str, grid: int) -> str:
    """The coupling implementation "auto" resolves to on a device, for a lattice size."""
    return "matmul" if grid <= DENSE_UP_TO.get(device_type, 16) else "fft"

__all__ = ["COUPLINGS", "KERNEL_SCALINGS", "PhaseBlock", "PhaseCore", "auto_impl", "ceiling_factor"]


def ceiling_factor(peak: torch.Tensor, ceiling: float, scaling: str) -> torch.Tensor:
    """The factor every weight of a channel's kernel is multiplied by, given its spectral peak.

    "exact" brings the peak to the ceiling, up or down; "cap" (paper 02) only
    brings it down. Where the peak exceeds the ceiling both return the same
    tensor, bit for bit: the cap is a clamp at 1 that does not bind.
    """
    if scaling not in KERNEL_SCALINGS:
        raise ValueError(f"unknown kernel scaling '{scaling}'")
    factor = ceiling / (peak + PEAK_EPS)
    return factor if scaling == "exact" else torch.clamp(factor, max=1.0)


class PhaseBlock(nn.Module):
    """C independent channels of G x G phase oscillators on the geometry `boundary`."""

    def __init__(self, channels: int, grid: int, dt: float = 0.1, coupling: str = "kuramoto",
                 damping: float = 0.0, spectral_clamp: float = 0.0, coupling_impl: str = "auto",
                 boundary: str = "torus", sakaguchi_alpha: float = 0.0, harmonic2_beta: float = 0.0,
                 kernel_scaling: str = "exact"):
        super().__init__()
        if coupling not in COUPLINGS:
            raise ValueError(f"unknown coupling '{coupling}'")
        if boundary not in BOUNDARIES:
            raise ValueError(f"unknown boundary '{boundary}'")
        if kernel_scaling not in KERNEL_SCALINGS:
            raise ValueError(f"unknown kernel scaling '{kernel_scaling}'")
        self.channels, self.grid, self.dt, self.coupling = channels, grid, dt, coupling
        self.damping, self.spectral_clamp, self.kernel_scaling = damping, spectral_clamp, kernel_scaling
        self.boundary = boundary
        self.geometry = build_geometry(boundary, grid)
        self._winfree_dc = None   # exact K*1 on the open geometries, set by prepare_coupling()

        if coupling == "sakaguchi":
            self.coupling_alpha = nn.Parameter(torch.full((channels, 1, 1), float(sakaguchi_alpha)))
        if coupling == "harmonic2":
            self.coupling_beta = nn.Parameter(torch.full((channels, 1, 1), float(harmonic2_beta)))
        if coupling == "winfree":
            self.winfree_s = nn.Parameter(torch.tensor([[-1.0, 0.0]]).repeat(channels, 1))
            self.winfree_i = nn.Parameter(torch.tensor([[1.0, 0.0, 1.0]]).repeat(channels, 1))

        # FFT and the dense circulant matmul are the same operator; "auto"
        # resolves at the first prepare_coupling() from the device the block
        # then lives on (`auto_impl`), and the dense operator's gather index is
        # built then, on that device, and only if it is used.
        if coupling_impl not in ("fft", "matmul", "auto"):
            raise ValueError(f"unknown coupling_impl '{coupling_impl}'")
        self.coupling_impl = coupling_impl
        self._circ: tuple[torch.device, torch.Tensor] | None = None

        # the draws, in paper 02's order: the kernels, then the natural frequencies
        self.kernel = nn.Parameter(torch.randn(channels, grid, grid) * 0.05)
        self.natural_freqs = nn.Parameter(torch.randn(channels, grid, grid) * 0.1 + 1.0)

    def _circ_idx(self) -> torch.Tensor:
        """The dense operator's gather index, built once on the kernel's device."""
        device = self.kernel.device
        if self._circ is None or self._circ[0] != device:
            idx = self.geometry.circulant_index()
            self._circ = (device, idx.reshape(-1, *idx.shape[-2:]).squeeze(0).to(device))
        return self._circ[1]

    def spectrum(self) -> torch.Tensor:
        """Each channel's kernel spectrum, before it meets the ceiling."""
        return self.geometry.kernel_spectrum(self.geometry.embed_kernel(self.kernel))

    def peaks(self) -> torch.Tensor:
        """[C]: each channel's spectral peak before scaling, the largest gain the raw kernel applies."""
        return self.spectrum().abs().flatten(1).amax(dim=1)

    def prepare_coupling(self) -> torch.Tensor:
        """Build the coupling operator once per scan: the complex spectrum (fft) or the dense matrix."""
        if self.coupling_impl == "auto":
            self.coupling_impl = auto_impl(self.kernel.device.type, self.grid)
        geom = self.geometry
        embedded = geom.embed_kernel(self.kernel)
        kfft = geom.kernel_spectrum(embedded)
        if self.spectral_clamp:
            mag = kfft.abs().flatten(1).amax(dim=1).view(-1, *([1] * (kfft.dim() - 1)))
            kfft = kfft * ceiling_factor(mag, self.spectral_clamp, self.kernel_scaling)

        if self.coupling_impl == "fft":
            coup: torch.Tensor = kfft
        else:
            taps = geom.spectrum_to_taps(kfft, embedded.shape)
            coup = geom.dense_operator(taps, self._circ_idx())

        if self.coupling == "winfree" and self.boundary not in ("torus", "cylinder"):
            # Winfree's influence term K*1 varies across the lattice on the open
            # and weighted geometries: computed exactly, once per scan
            ones = self.kernel.new_ones(1, 1, self.channels, self.grid, self.grid)
            self._winfree_dc = self._apply_conv(ones, coup)[0, 0]  # [C, G, G]
        return coup

    def _apply_conv(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        """Apply the coupling operator to stacked fields [B, F, C, G, G]."""
        if not coup.is_complex():
            x = field.flatten(-2)  # [B, F, C, N]
            return torch.einsum("bxcn,cmn->bxcm", x, coup).view_as(field)
        return self.geometry.apply(field, coup)

    def step(self, theta: torch.Tensor, drive: torch.Tensor, coup: torch.Tensor, substeps: int,
             drive_quad: torch.Tensor | None = None) -> torch.Tensor:
        """Advance phases. theta, drive: [B, C, G, G]; drive_quad [B, C, G, G, 2] or None."""
        om = self.natural_freqs
        if self.coupling == "winfree":
            if self.boundary in ("torus", "cylinder"):
                # the per-channel kernel sum is the operator's DC response
                ksum = (coup[:, 0, 0].real if coup.is_complex() else coup.sum(dim=-1)[:, 0])
                ksum = ksum.view(-1, 1, 1)
            else:
                ksum = self._winfree_dc  # [C, G, G], exact spatial K*1
        for _ in range(substeps):
            s, c = torch.sin(theta), torch.cos(theta)
            fields = [s, c]
            if self.coupling == "harmonic2":  # sin/cos(2 theta) ride the same transform
                fields += [torch.sin(2 * theta), torch.cos(2 * theta)]
            conv = self._apply_conv(torch.stack(fields, dim=1), coup)  # [B, F, C, G, G]
            conv_sin, conv_cos = conv[:, 0], conv[:, 1]
            if self.coupling == "kuramoto":
                torque = c * conv_sin - s * conv_cos
            elif self.coupling == "winfree":
                sens = self.winfree_s[:, 0:1, None] * s + self.winfree_s[:, 1:2, None] * c
                infl = (self.winfree_i[:, 0:1, None] * ksum
                        + self.winfree_i[:, 1:2, None] * conv_sin
                        + self.winfree_i[:, 2:3, None] * conv_cos)
                torque = sens * infl
            elif self.coupling == "sakaguchi":
                # sin(D - a) = sin D cos a - cos D sin a, from the same two convolutions
                base = c * conv_sin - s * conv_cos
                quad = c * conv_cos + s * conv_sin
                torque = (torch.cos(self.coupling_alpha) * base - torch.sin(self.coupling_alpha) * quad)
            else:  # harmonic2: Kuramoto + beta * the second-harmonic term
                s2, c2 = torch.sin(2 * theta), torch.cos(2 * theta)
                torque = (c * conv_sin - s * conv_cos
                          + self.coupling_beta * (c2 * conv[:, 2] - s2 * conv[:, 3]))
            if self.damping:
                torque = torque - self.damping * s  # the restoring pull: sin(0 - theta) = -sin(theta)
            if drive_quad is not None:
                torque = torque + drive_quad[..., 1] * c - drive_quad[..., 0] * s
            theta = theta + self.dt * (om + torque + drive)
            theta = torch.remainder(theta, TWO_PI)
        return theta


class PhaseCore(nn.Module):
    """One PhaseBlock, scanned over frames. State is the phase field [B, 1, C, G, G];
    the signals are [sin theta, cos theta], flattened."""

    def __init__(self, channels: int = 4, grid: int = 16, blocks: int = 1, substeps: int = 1,
                 dt: float = 0.1, coupling: str = "kuramoto", damping: float = 0.0,
                 spectral_clamp: float = 0.0, coupling_impl: str = "auto", seed: int = 42,
                 boundary: str = "torus", sakaguchi_alpha: float = 0.0, harmonic2_beta: float = 0.0,
                 kernel_scaling: str = "exact"):
        super().__init__()
        if blocks != 1:
            raise ValueError("paper 03 runs one block of coupled oscillators")
        self.channels, self.grid, self.substeps = channels, grid, substeps
        self.boundary = boundary
        self.blocks = nn.ModuleList([PhaseBlock(channels, grid, dt, coupling, damping, spectral_clamp,
                                                coupling_impl, boundary, sakaguchi_alpha=sakaguchi_alpha,
                                                harmonic2_beta=harmonic2_beta, kernel_scaling=kernel_scaling)])
        gen = torch.Generator().manual_seed(seed)
        phase0 = torch.rand(1, 1, channels, grid, grid, generator=gen) * TWO_PI
        self.register_buffer("phase0", phase0)   # fixed seeded start: deterministic, symmetry broken

    @property
    def readout_dim(self) -> int:
        return 2 * self.channels * self.grid * self.grid

    def init_state(self, batch: int) -> torch.Tensor:
        return self.phase0.expand(batch, -1, -1, -1, -1).clone()

    def forward_scan(self, drives: torch.Tensor, state: torch.Tensor | None = None,
                     drives_quad: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """drives [B, T, C, G, G] (and drives_quad [B, T, C, G, G, 2]) -> (signals [B, T, 2CGG], final state)."""
        b = drives.shape[0]
        state = self.init_state(b) if state is None else state
        blk = self.blocks[0]
        coup = blk.prepare_coupling()
        theta = state[:, 0]
        frames = drives.unbind(1)
        qframes = drives_quad.unbind(1) if drives_quad is not None else (None,) * len(frames)
        outs = []
        for drive_t, quad_t in zip(frames, qframes, strict=True):
            theta = blk.step(theta, drive_t, coup, self.substeps, drive_quad=quad_t)
            outs.append(torch.cat((torch.sin(theta), torch.cos(theta)), dim=1).flatten(1))
        return torch.stack(outs, dim=1), theta[:, None]

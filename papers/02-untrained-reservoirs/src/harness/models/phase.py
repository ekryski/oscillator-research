"""The phase-oscillator core: a population of coupled phases on a lattice.

C channels of G x G phase oscillators. Coupling is translation-invariant — one
kernel of taps applied at every site — so it is a convolution, computed in
O(N log N) by FFT or as one dense circulant matmul, whichever the device
prefers. The lattice's edge gluing is a separate concern entirely and lives in
`harness.models.geometries`; this module is geometry-agnostic.

Physics per Euler substep (coupling="kuramoto", the default):

    torque_i = sum_j K(i - j) * sin(theta_j - theta_i)
             = cos(theta_i) * (K * sin theta)_i  -  sin(theta_i) * (K * cos theta)_i

    theta <- theta + dt * (omega + torque + drive)

The relative-phase term (theta_j - theta_i) is what produces synchronization
pressure — oscillators pulling each other toward agreement. The ablation
coupling="forced" computes only (K * sin theta), a linear forcing with no such
pressure, which is how the paper separates "the coupling did something" from
"a linear filter did something".

Coupling-law variants. Each reduces EXACTLY to kuramoto at its zero init, so a
trained arm starts warm-comparable — but that also makes zero init degenerate
when FROZEN, which is why the untrained matrices must pass explicit nonzero
canonical values or the family is not a distinct factor level at all:

    sakaguchi   sum_j K sin(theta_j - theta_i - alpha): a per-channel phase lag
                that breaks pure-attraction symmetry and admits travelling waves
    harmonic2   + beta * sum_j K sin(2(theta_j - theta_i)): the cluster-forming
                second harmonic, i.e. multi-group states
    winfree     separable S(theta) * (K * I(theta)) with low-order-harmonic
                sensitivity and influence, at the classic S = -sin, I = 1 + cos

Damping > 0 adds a pacemaker term -damping*sin(theta): coupling to a grounding
oscillator at phase 0. Without it the dynamics are marginally stable and BPTT
gradients through long scans explode (observed: grad norms in the thousands).
The pacemaker gives a contraction toward a driven regime — an echo-state-style
bounded memory horizon.

Quadrature Adler drive. Besides the additive drive above, the step functions
accept an OPTIONAL second drive tensor holding a quadrature pair
(qcos, qsin) = gain * A * (cos phi, sin phi) per site. It injects the
phase-referenced torque computed from the CURRENT theta at each substep:

    torque_quad = gain*A*sin(phi - theta) = qsin*cos(theta) - qcos*sin(theta)

i.e. genuine carrier-relative entrainment rather than a phase-blind push. When
it is None every code path is bit-identical to the additive-only step; frozen
controls depend on that.

Spectral clamp. The clamp rescales each channel so max |K-hat| stays under the
cap. For the fully periodic venues that IS the operator's exact 2-norm; for the
padded and weighted ones, pad/crop/weighting all have norm <= 1, so it remains
a true bound. The twisted pair needs a correction and supplies its own
`clamp_factor` — see `geometries/twisted.py`.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from harness.models.geometries import BOUNDARIES, build_geometry

TWO_PI = 2 * math.pi
COUPLINGS = ("kuramoto", "forced", "sakaguchi", "harmonic2", "winfree")

__all__ = ["COUPLINGS", "PhaseBlock", "PhaseCore"]


class PhaseBlock(nn.Module):
    """One field block: C independent channels of G x G phase oscillators on
    the venue selected by `boundary`."""

    def __init__(
        self,
        channels: int,
        grid: int,
        dt: float = 0.1,
        coupling: str = "kuramoto",
        damping: float = 0.0,
        spectral_clamp: float = 0.0,
        coupling_impl: str = "auto",
        kernel_support: int = 0,
        boundary: str = "torus",
        damping_learnable: bool = False,
        sakaguchi_alpha: float = 0.0,
        harmonic2_beta: float = 0.0,
    ):
        super().__init__()
        if coupling not in COUPLINGS:
            raise ValueError(f"unknown coupling '{coupling}'")
        if boundary not in BOUNDARIES:
            raise ValueError(f"unknown boundary '{boundary}'")
        self.channels, self.grid, self.dt, self.coupling = channels, grid, dt, coupling
        self.damping = damping
        self.spectral_clamp = spectral_clamp
        self.boundary = boundary
        self.geometry = build_geometry(boundary, grid)
        # exact K*1 DC response, needed by winfree where it is not constant;
        # set by prepare_coupling()
        self._winfree_dc = None

        if damping_learnable:
            # per-channel learnable pinning (raw param, init at the scalar;
            # a multi-timescale spread is what training may discover)
            self.damping_lam = nn.Parameter(torch.full((channels, 1, 1), float(damping)))
        if coupling == "sakaguchi":
            self.coupling_alpha = nn.Parameter(
                torch.full((channels, 1, 1), float(sakaguchi_alpha)))
        if coupling == "harmonic2":
            self.coupling_beta = nn.Parameter(
                torch.full((channels, 1, 1), float(harmonic2_beta)))
        if coupling == "winfree":
            self.winfree_s = nn.Parameter(torch.tensor([[-1.0, 0.0]]).repeat(channels, 1))
            self.winfree_i = nn.Parameter(torch.tensor([[1.0, 0.0, 1.0]]).repeat(channels, 1))

        # kernel_support > 0 restricts K to wrapped offsets within that
        # Chebyshev radius (nearest-neighbour bonds at 1) — the local-coupling
        # ablation. The mask is fixed; training sees only the surviving taps.
        if kernel_support:
            if not self.geometry.supports_kernel_support:
                raise ValueError(
                    f"kernel_support is defined on the 2-D row/column offset grid; "
                    f"the {boundary} venue indexes offsets differently")
            off = torch.arange(grid)
            d = torch.minimum(off, grid - off)  # wrapped 1-D distance per axis
            mask = ((d[:, None] <= kernel_support) & (d[None, :] <= kernel_support)).float()
            self.register_buffer("kernel_mask", mask.expand(channels, grid, grid).clone(),
                                 persistent=False)
        else:
            self.kernel_mask = None

        # FFT and dense matmul are numerically identical operators with very
        # different speed per device (measured, batch-24/8 x T=350 fwd+bwd:
        # CUDA 3090 fft 391 ms vs matmul 457 ms; MPS fft 835 ms vs matmul
        # 267 ms). "auto" resolves lazily at the first prepare_coupling() from
        # the device the module actually lives on; large grids always use fft,
        # because the circulant is N^2.
        if coupling_impl not in ("fft", "matmul", "auto"):
            raise ValueError(f"unknown coupling_impl '{coupling_impl}'")
        if coupling_impl == "auto" and grid > 32:
            coupling_impl = "fft"
        self.coupling_impl = coupling_impl

        self.kernel = nn.Parameter(torch.randn(channels, grid, grid) * 0.05)
        self.natural_freqs = nn.Parameter(torch.randn(channels, grid, grid) * 0.1 + 1.0)
        if coupling_impl in ("matmul", "auto"):
            idx = self.geometry.circulant_index()
            self.register_buffer("_circ_idx", idx.reshape(-1, *idx.shape[-2:]).squeeze(0),
                                 persistent=False)

    def prepare_coupling(self) -> torch.Tensor:
        """Build the coupling operator once per scan rather than per substep.

        Returns whatever `apply` consumes: the complex spectrum (fft impl) or
        the dense [C, N, N] matrix (matmul impl). Mathematically the same
        operator either way.
        """
        if self.coupling_impl == "auto":  # resolve once, from the real device
            self.coupling_impl = "fft" if self.kernel.device.type == "cuda" else "matmul"
        geom = self.geometry
        kernel = self.kernel if self.kernel_mask is None else self.kernel * self.kernel_mask
        embedded = geom.embed_kernel(kernel)
        kfft = geom.kernel_spectrum(embedded)

        if self.spectral_clamp:
            cap = self.spectral_clamp / geom.clamp_factor
            mag = kfft.abs().flatten(1).amax(dim=1).view(-1, *([1] * (kfft.dim() - 1)))
            kfft = kfft * torch.clamp(cap / (mag + 1e-8), max=1.0)

        if self.coupling_impl == "fft":
            coup: torch.Tensor = kfft
        else:
            taps = geom.spectrum_to_taps(kfft, embedded.shape)
            coup = geom.dense_operator(taps, self._circ_idx)

        if self.coupling == "winfree" and self.boundary not in ("torus", "cylinder"):
            # Winfree's influence DC term K*1 is spatially VARYING on open or
            # metric-weighted venues — compute it exactly, once per scan.
            # (torus/cylinder keep their scalar shortcut in step(), which is
            # bit-identical there; frozen controls depend on it.)
            ones = self.kernel.new_ones(1, 1, self.channels, self.grid, self.grid)
            self._winfree_dc = self._apply_conv(ones, coup)[0, 0]  # [C, G, G]
        return coup

    def _apply_conv(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        """Apply the coupling operator to stacked fields [B, F, C, G, G].

        The dense path is geometry-blind: the venue already lives inside the
        matrix. The spectral path hands off to the geometry.
        """
        if not coup.is_complex():
            x = field.flatten(-2)  # [B, F, C, N]
            return torch.einsum("bxcn,cmn->bxcm", x, coup).view_as(field)
        return self.geometry.apply(field, coup)

    def step(self, theta: torch.Tensor, drive: torch.Tensor, coup: torch.Tensor, substeps: int,
             omega: torch.Tensor | None = None,
             drive_quad: torch.Tensor | None = None) -> torch.Tensor:
        """Advance phases. theta, drive: [B, C, G, G]; coup from
        prepare_coupling(). omega overrides natural_freqs when given (the
        input-as-omega arm; may be per-example). drive_quad [B, C, G, G, 2]
        adds the Adler torque from the CURRENT theta each substep; None keeps
        the additive-only path bit-identical."""
        om = self.natural_freqs if omega is None else omega
        if self.coupling == "winfree":
            if self.boundary in ("torus", "cylinder"):
                # per-channel kernel sum = the operator's DC response
                ksum = (coup[:, 0, 0].real if coup.is_complex() else coup.sum(dim=-1)[:, 0])
                ksum = ksum.view(-1, 1, 1)
            else:
                ksum = self._winfree_dc  # [C, G, G], exact spatial K*1
        for _ in range(substeps):
            s, c = torch.sin(theta), torch.cos(theta)
            fields = [s, c]
            if self.coupling == "harmonic2":  # sin/cos(2t) ride the same transform
                fields += [torch.sin(2 * theta), torch.cos(2 * theta)]
            conv = self._apply_conv(torch.stack(fields, dim=1), coup)  # [B, F, C, G, G]
            conv_sin, conv_cos = conv[:, 0], conv[:, 1]
            if self.coupling == "kuramoto":
                torque = c * conv_sin - s * conv_cos
            elif self.coupling == "forced":  # linear-forcing ablation K * sin(theta)
                torque = conv_sin
            elif self.coupling == "winfree":
                sens = self.winfree_s[:, 0:1, None] * s + self.winfree_s[:, 1:2, None] * c
                infl = (self.winfree_i[:, 0:1, None] * ksum
                        + self.winfree_i[:, 1:2, None] * conv_sin
                        + self.winfree_i[:, 2:3, None] * conv_cos)
                torque = sens * infl
            elif self.coupling == "sakaguchi":
                # sin(D-a) = sinD cosa - cosD sina, with sum_j K cos(theta_j-theta_i)
                # = cos(K*cos) + sin(K*sin) from the same two convolutions.
                base = c * conv_sin - s * conv_cos
                quad = c * conv_cos + s * conv_sin
                torque = (torch.cos(self.coupling_alpha) * base
                          - torch.sin(self.coupling_alpha) * quad)
            else:  # harmonic2: kuramoto + beta * the second-harmonic term
                s2, c2 = torch.sin(2 * theta), torch.cos(2 * theta)
                torque = (c * conv_sin - s * conv_cos
                          + self.coupling_beta * (c2 * conv[:, 2] - s2 * conv[:, 3]))
            lam = self.damping_lam if hasattr(self, "damping_lam") else self.damping
            if lam is not None and (torch.is_tensor(lam) or lam):
                torque = torque - lam * s  # pacemaker: sin(0 - theta) = -sin(theta)
            if drive_quad is not None:
                torque = torque + drive_quad[..., 1] * c - drive_quad[..., 0] * s
            theta = theta + self.dt * (om + torque + drive)
            theta = torch.remainder(theta, TWO_PI)
        return theta


class PhaseCore(nn.Module):
    """A stack of PhaseBlocks with 1x1 channel mixing between them.

    State is the full phase field [B, blocks, C, G, G] and persists across
    frames — this is the model's recurrent memory. Readout is
    [sin(theta), cos(theta)] of the last block, flattened.
    """

    def __init__(
        self,
        channels: int = 32,
        grid: int = 16,
        blocks: int = 1,
        substeps: int = 2,
        dt: float = 0.1,
        coupling: str = "kuramoto",
        damping: float = 0.0,
        tbptt: int = 0,
        spectral_clamp: float = 0.0,
        coupling_impl: str = "auto",
        compile_step: bool = False,
        seed: int = 42,
        kernel_support: int = 0,
        boundary: str = "torus",
        damping_learnable: bool = False,
        sakaguchi_alpha: float = 0.0,
        harmonic2_beta: float = 0.0,
    ):
        super().__init__()
        self.channels, self.grid, self.blocks_n, self.substeps = channels, grid, blocks, substeps
        self.boundary = boundary
        self.tbptt = tbptt  # detach state every N frames while training (0 = full BPTT)
        self.blocks = nn.ModuleList(
            PhaseBlock(channels, grid, dt, coupling, damping, spectral_clamp, coupling_impl,
                       kernel_support, boundary, damping_learnable,
                       sakaguchi_alpha=sakaguchi_alpha, harmonic2_beta=harmonic2_beta)
            for _ in range(blocks)
        )
        # approximation honesty: the sphere is a lattice approximation, not
        # true S^2 harmonics — exposed so runs and logs can record which
        self.sphere_impl = self.blocks[0].geometry.implementation if boundary == "sphere" else None
        if compile_step:
            # removes per-op Python/autograd dispatch — the measured bottleneck
            for blk in self.blocks:
                blk.step = torch.compile(blk.step, dynamic=False)
        # sin/cos of the previous block feed the next block's drive
        self.mixers = nn.ModuleList(nn.Conv2d(2 * channels, channels, 1) for _ in range(blocks - 1))
        gen = torch.Generator().manual_seed(seed)
        phase0 = torch.rand(1, blocks, channels, grid, grid, generator=gen) * TWO_PI
        # fixed seeded init: deterministic eval, symmetry broken
        self.register_buffer("phase0", phase0)

    @property
    def readout_dim(self) -> int:
        return 2 * self.channels * self.grid * self.grid

    def init_state(self, batch: int) -> torch.Tensor:
        return self.phase0.expand(batch, -1, -1, -1, -1).clone()

    def prepare_couplings(self) -> list[torch.Tensor]:
        return [b.prepare_coupling() for b in self.blocks]

    def readout(self, state: torch.Tensor) -> torch.Tensor:
        """state [B, blocks, C, G, G] -> [B, 2*C*G*G] from the last block."""
        theta = state[:, -1]
        return torch.cat((torch.sin(theta), torch.cos(theta)), dim=1).flatten(1)

    def step_frame(
        self, state: torch.Tensor, drive: torch.Tensor, coups: list[torch.Tensor] | None = None,
        omega_override: torch.Tensor | None = None, drive_quad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One frame update. drive [B, C, G, G] and drive_quad both enter the
        FIRST block only — external input reaches the field once.
        Returns (new_state, readout [B, 2*C*G*G])."""
        if coups is None:
            coups = self.prepare_couplings()
        new_blocks = []
        block_in = drive
        for i, blk in enumerate(self.blocks):
            theta = blk.step(state[:, i], block_in, coups[i], self.substeps,
                             omega=omega_override if i == 0 else None,
                             drive_quad=drive_quad if i == 0 else None)
            new_blocks.append(theta)
            if i + 1 < self.blocks_n:
                feats = torch.cat((torch.sin(theta), torch.cos(theta)), dim=1)  # [B, 2C, G, G]
                block_in = self.mixers[i](feats)
        new_state = torch.stack(new_blocks, dim=1)
        return new_state, self.readout(new_state)

    def forward_scan(
        self, drives: torch.Tensor, state: torch.Tensor | None = None, grad_ckpt: int = 0,
        omega_override: torch.Tensor | None = None, drives_quad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scan over a frame sequence. drives [B, T, C, G, G]; drives_quad
        [B, T, C, G, G, 2] optionally adds the per-frame Adler drive.
        Returns (features [B, T, readout_dim], final_state).

        grad_ckpt > 0 recomputes activations in segments of that many frames
        during backward (~2x compute for ~segment-fold less activation memory).
        """
        b, t = drives.shape[:2]
        if state is None:
            state = self.init_state(b)
        coups = self.prepare_couplings()

        def scan_segment(state_in: torch.Tensor, seg: torch.Tensor,
                         seg_quad: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
            outs = []
            st = state_in
            grad_on = torch.is_grad_enabled()
            if grad_on and self.training and not st.requires_grad:
                # Uniform requires_grad across ALL frames: torch.compile guards
                # on the flag, and a flipping flag forces a recompile per flip
                # until dynamo's limit trips and everything falls back to
                # eager-with-overhead. requires_grad_ on the detached copy keeps
                # the gradient CUT identical while keeping the signature stable.
                st = st.detach().requires_grad_(True)
            # unbind, don't slice: seg[:, i] inside the loop makes autograd
            # accumulate a FULL [B,T,C,G,G] gradient per frame in backward
            # (measured: ~1.9 s/step of pure memory traffic, 90% of the step).
            # unbind's backward is one stack over all frame grads instead.
            frames = seg.unbind(1)
            qframes = seg_quad.unbind(1) if seg_quad is not None else (None,) * len(frames)
            for i, drive_t in enumerate(frames):
                if self.training and self.tbptt and i and i % self.tbptt == 0:
                    st = st.detach()  # truncate the gradient path, keep the state
                    if grad_on:
                        st = st.requires_grad_(True)
                st, feat = self.step_frame(st, drive_t, coups, omega_override,
                                           drive_quad=qframes[i])
                outs.append(feat)
            return st, torch.stack(outs, dim=1)

        if grad_ckpt and self.training:
            feats_all = []
            for start in range(0, t, grad_ckpt):
                seg = drives[:, start : start + grad_ckpt]
                seg_q = None if drives_quad is None else drives_quad[:, start : start + grad_ckpt]
                state, feats = checkpoint(scan_segment, state, seg, seg_q, use_reentrant=False)
                feats_all.append(feats)
            return torch.cat(feats_all, dim=1), state
        state, feats = scan_segment(state, drives, drives_quad)
        return feats, state

"""`OscillatorField` — the thing under test.

A core (phase, Stuart-Landau, or random-graph) wrapped in the fixed tonotopic
injection and a FROZEN random readout probe. The physics-only invariant is
asserted at construction: nothing outside the physics parameter name set is
ever trainable, so a trained gain cannot come from scaffolding.

The probe is a buffer rather than a parameter for the same reason. Gradients
have to reshape the DYNAMICS until a random projection of the field's summary
statistics separates classes; they are not allowed to fix up the reader.

Feature contract, shared by every arm so the ridge comparison stays fair:

    features           whole-clip pooled statistics
    features_windowed  the same, per quarter-window — the temporal profile the
                       whole-clip means erase
    features_macro     mesoscale organization: local order parameter per patch,
                       plus per-row amplitude
    features_settle    a drive-free continuation after the clip, featurized
                       alone — persistence with the stimulus removed
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from harness.measurement.probe import phase_features
from harness.models.phase import PhaseCore
from harness.models.random_graph import RandGraphCore
from harness.models.stuart_landau import SLCore
from harness.stimuli.filterbank import band_edges
from harness.stimuli.injection import quad_rows_to_drive, rows_to_drive
from harness.utils.constants import GAIN, N_SETTLE, PROBE_SCALE, TWO_PI, WARMUP_FRAMES


def physics_block(core) -> object:
    """The module owning kernel/natural_freqs (PhaseCore block 0, or SLCore itself)."""
    return core.blocks[0] if hasattr(core, "blocks") else core


class OscillatorField(nn.Module):
    """PhaseCore + fixed tonotopic injection + frozen random probe.

    Trainable = physics only (kernel, natural_freqs) — asserted below. The probe
    is a buffer: gradients shape the DYNAMICS so that a random projection of the
    field's summary statistics separates classes."""

    def __init__(self, channels: int = 4, grid: int = 16, coupling: str = "kuramoto",
                 damping: float = 0.5, spectral_clamp: float = 0.5, substeps: int = 1,
                 dt: float = 0.1, n_classes: int = 8, probe_seed: int = 0,
                 gain: float = GAIN, seed: int = 42, blocks: int = 1,
                 kernel_support: int = 0, core: str = "phase", boundary: str = "torus",
                 damping_learnable: bool = False, omega_encoder: bool = False,
                 sakaguchi_alpha: float = 0.0, harmonic2_beta: float = 0.0,
                 graph_k: int = 1):
        super().__init__()
        self.channels, self.grid, self.gain = channels, grid, gain
        if core == "randgraph":
            assert not omega_encoder, "randgraph: omega_encoder unsupported"
            self.core = RandGraphCore(channels=channels, grid=grid, dt=dt,
                                      damping=damping, spectral_clamp=spectral_clamp,
                                      graph_k=graph_k, seed=seed, substeps=substeps)
        elif core == "phase":
            self.core = PhaseCore(channels=channels, grid=grid, blocks=blocks, substeps=substeps,
                                  dt=dt, coupling=coupling, damping=damping, tbptt=0,
                                  spectral_clamp=spectral_clamp, coupling_impl="auto",
                                  compile_step=False, seed=seed, kernel_support=kernel_support,
                                  boundary=boundary, damping_learnable=damping_learnable,
                                  sakaguchi_alpha=sakaguchi_alpha, harmonic2_beta=harmonic2_beta)
        else:  # "sl" (arm C) | "sl-fixedamp" (arm B) — Stuart-Landau
            assert blocks == 1 and kernel_support == 0, "SL core: blocks/support unsupported"
            assert boundary == "torus", "cylinder boundary: phase core only (scope)"
            self.core = SLCore(channels=channels, grid=grid, substeps=substeps, dt=dt,
                               damping=damping, spectral_clamp=spectral_clamp,
                               amplitude_frozen=(core == "sl-fixedamp"), seed=seed,
                               damping_learnable=damping_learnable)
        if blocks > 1:
            # Physics-only invariant: the 1x1 inter-block mixers are bookend-class
            # capacity — frozen at their random init (fixed routing), so stage-E
            # depth runs stay a pure physics comparison.
            for p in self.core.mixers.parameters():
                p.requires_grad_(False)
        # input-as-omega: a tiny tonotopic tuner (row-energy -> per-row
        # omega modulation; 272 params at G=16, zero-init => warm-equivalent to
        # frozen). Registered exception to physics-only: the TUNER is the
        # hypothesis under test — additive drive is OFF for this arm, so
        # omega-writing is the only input channel.
        self.omega_encoder = omega_encoder
        if omega_encoder:
            assert core == "phase", "omega encoder: phase core only"
            self.omega_enc_w = nn.Parameter(torch.zeros(grid, grid))
            self.omega_enc_b = nn.Parameter(torch.zeros(grid))
        self.feat_dim = 4 * channels * grid * grid
        pg = torch.Generator().manual_seed(probe_seed)
        probe = torch.randn(self.feat_dim, n_classes, generator=pg) / math.sqrt(self.feat_dim)
        self.register_buffer("probe_w", probe)
        # physics = kernel/omega (+ per-channel coupling-law / amplitude params);
        # invariant: NOTHING outside the physics name set is trainable (an
        # amplitude-frozen SL arm may freeze some physics params — that's fine,
        # the violation to catch is a trainable non-physics param).
        physics = ("kernel", "natural_freqs", "coupling_alpha", "coupling_beta",
                   "alpha", "beta_hat", "damping_lam", "winfree_s", "winfree_i",
                   "omega_enc_w", "omega_enc_b")
        rogue = sorted(n for n, p in self.named_parameters() if p.requires_grad
                       and not any(n.endswith(t) for t in physics))
        assert not rogue, f"physics-only invariant violated: trainable non-physics {rogue}"

    def _drives(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """rows [B,T,G] -> (additive drives, None); quadrature rows [B,T,G,2]
        (quad frontend) -> (zero additive drives, Adler quad drives).
        Both drives are broadcast views — no materialized copies."""
        if rows.dim() == 4:
            assert not self.omega_encoder, "quad frontend: omega-encoder arm undefined"
            assert hasattr(self.core, "blocks"), \
                "quad frontend: phase core only (SL Adler hook documented in torus.py, not built)"
            b, t, g, _ = rows.shape
            zeros = rows.new_zeros(()).expand(b, t, self.channels, g, g)
            return zeros, quad_rows_to_drive(rows, self.channels, self.gain)
        return rows_to_drive(rows, self.channels, self.gain), None

    def _scan_full(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The scan behind every featurizer: (features [B,T,D], final state)."""
        if self.omega_encoder:
            mod = torch.tanh(rows.abs().mean(dim=1) @ self.omega_enc_w.T + self.omega_enc_b)
            om = self.core.blocks[0].natural_freqs[None] * (1 + mod[:, None, :, None])
            drives = torch.zeros(rows.shape[0], rows.shape[1], self.channels,
                                 self.grid, self.grid, device=rows.device)
            return self.core.forward_scan(drives, omega_override=om)
        drives, quad = self._drives(rows)
        if quad is not None:
            return self.core.forward_scan(drives, drives_quad=quad)
        # historical call kept verbatim (SL cores lack the quad kwarg)
        return self.core.forward_scan(drives)

    def _scan(self, rows: torch.Tensor) -> torch.Tensor:
        return self._scan_full(rows)[0]

    def features(self, rows: torch.Tensor,
                 tvalid: torch.Tensor | None = None) -> torch.Tensor:
        return phase_features(self._scan(rows), self.channels, self.grid,
                              tvalid=tvalid)

    def features_windowed(self, rows: torch.Tensor, windows: int = 4,
                          tvalid: torch.Tensor | None = None) -> torch.Tensor:
        """Additional metric: per-window pooled features (windows x feat_dim)
        — lets the ridge see the temporal PROFILE the whole-clip means erase.
        Reported as an ADDITIONAL metric; the standard metric is unchanged.

        tvalid (length masking): per-clip windows split the VALID range
        [WARMUP, tvalid_i) instead of the padded clip. Clips too short for
        2-frame windows extend hi into the ring-down only as far as needed
        (hi >= WARMUP + 2*windows, clamped to T) — deterministic, identical
        rule for every arm; tvalid == T reproduces the unmasked windows
        exactly. Per-clip loop — eval-only, acceptable."""
        if tvalid is None:
            feats = self._scan(rows)[:, WARMUP_FRAMES:]  # settle once, globally
            t = feats.shape[1]
            chunks = [phase_features(feats[:, i * t // windows:(i + 1) * t // windows],
                                     self.channels, self.grid, warmup=1)
                      for i in range(windows)]
            return torch.cat(chunks, dim=1)
        feats = self._scan(rows)
        t = feats.shape[1]
        lo = WARMUP_FRAMES
        out = []
        for i in range(feats.shape[0]):
            hi = min(t, max(int(tvalid[i]), lo + 2 * windows))
            chunks = [phase_features(
                feats[i:i + 1, lo + (hi - lo) * j // windows: lo + (hi - lo) * (j + 1) // windows],
                self.channels, self.grid, warmup=1) for j in range(windows)]
            out.append(torch.cat(chunks, dim=1))
        return torch.cat(out, dim=0)

    def features_macro(self, rows: torch.Tensor, patch: int = 4,
                       tvalid: torch.Tensor | None = None) -> torch.Tensor:
        """Macroscopic readout: mesoscale organization the pooled
        trajectories never expose. Two blocks, time mean+std each:
        (1) local Kuramoto order parameter R on non-overlapping patch x patch
        neighborhoods per channel (chimera geometry — which regions cohere);
        (2) per-row |z| (amplitude signaling; identically 1 for phase cores).
        No new parameters — physics-only invariant untouched.

        tvalid (length masking): time mean/std over valid frames
        [WARMUP, tvalid_i) only (>= 2 frames — the std floor; tvalid == T
        reproduces the unmasked statistics exactly)."""
        assert self.grid % patch == 0, f"grid {self.grid} not divisible by patch {patch}"
        feats = self._scan(rows)[:, WARMUP_FRAMES:]
        b, t, _ = feats.shape
        f = feats.view(b, t, 2 * self.channels, self.grid, self.grid)
        y, x = f[:, :, :self.channels], f[:, :, self.channels:]
        amp = torch.sqrt(x * x + y * y).clamp_min(1e-8)
        cs, sn = (x / amp).flatten(0, 1), (y / amp).flatten(0, 1)  # unit phasors [B*T,C,G,G]
        rloc = torch.sqrt(F.avg_pool2d(cs, patch) ** 2 + F.avg_pool2d(sn, patch) ** 2)
        rloc = rloc.view(b, t, -1)                     # [B, T, C*(G/patch)^2]
        row_amp = amp.mean(dim=4).view(b, t, -1)       # [B, T, C*G]
        if tvalid is None:
            return torch.cat((rloc.mean(dim=1), rloc.std(dim=1),
                              row_amp.mean(dim=1), row_amp.std(dim=1)), dim=1)
        tv = (tvalid.to(feats.device) - WARMUP_FRAMES).clamp(2, t)
        m = (torch.arange(t, device=feats.device)[None, :] < tv[:, None])[:, :, None].float()
        n = m.sum(dim=1)

        def masked_stats(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mean = (z * m).sum(dim=1) / n
            var = ((z - mean[:, None]) ** 2 * m).sum(dim=1) / (n - 1).clamp_min(1.0)
            return mean, var.sqrt()

        rl_m, rl_s = masked_stats(rloc)
        ra_m, ra_s = masked_stats(row_amp)
        return torch.cat((rl_m, rl_s, ra_m, ra_s), dim=1)

    def features_settle(self, rows: torch.Tensor, tvalid: torch.Tensor | None = None,
                        n_settle: int = N_SETTLE) -> torch.Tensor:
        """Settle-read column: drive the field over
        the full clip, then continue n_settle DRIVE-FREE frames from the final
        state and featurize that settle window ALONE (warmup=1 — the window is
        its own regime; feat_dim contract unchanged). tvalid is accepted for
        call-site symmetry but unused by design: every settle frame is valid,
        and the driven scan covers the identical padded clip for every arm
        (digit padding is already near-silent drive; a per-clip tvalid-anchored
        settle start would be a different pre-registered experiment).
        omega-encoder arms settle at their NATURAL omega — the encoder's
        override is input-derived, and the settle regime is stimulus-removed."""
        _, state = self._scan_full(rows)
        zeros = rows.new_zeros(()).expand(rows.shape[0], n_settle, self.channels,
                                          self.grid, self.grid)
        feats, _ = self.core.forward_scan(zeros, state=state)
        return phase_features(feats, self.channels, self.grid, warmup=1)

    def forward(self, rows: torch.Tensor,
                tvalid: torch.Tensor | None = None) -> torch.Tensor:
        return self.features(rows, tvalid) @ self.probe_w * PROBE_SCALE

    @torch.no_grad()
    def phase_trajectory(self, rows: torch.Tensor) -> torch.Tensor:
        """[B,T,G] (or quad [B,T,G,2]) -> theta [B,T,C,G,G] for instrumentation."""
        drives, quad = self._drives(rows)
        if quad is not None:
            feats, _ = self.core.forward_scan(drives, drives_quad=quad)
        else:
            feats, _ = self.core.forward_scan(drives)
        b, t, _ = feats.shape
        f = feats.view(b, t, 2 * self.channels, self.grid, self.grid)
        return torch.atan2(f[:, :, :self.channels], f[:, :, self.channels:])


def tonotopic_omega(channels: int, grid: int, dt: float, substeps: int,
                    gen: torch.Generator, jitter: float = 0.05) -> torch.Tensor:
    """The engineered-cochlea init — "design the resonator bank": row r's
    oscillators get the natural frequency whose free rotation matches the
    filterbank band-r center, with small per-oscillator jitter — coverage by
    design instead of asking gradients to cross tongue deserts. Rows whose
    required rate sits below the pinning barrier come out pinned-static at that
    lambda; that dead-resonator cost is part of the design, recorded per run."""
    e = band_edges(grid)
    centers = torch.sqrt(e[:-1] * e[1:])  # [G] band centers, cycles/frame
    theta_dot = (TWO_PI * centers / (dt * substeps)).to(torch.float32)
    base = theta_dot.view(1, grid, 1).expand(channels, grid, grid)
    return base * (1 + jitter * torch.randn(channels, grid, grid, generator=gen))


def shuffle_kernel_(model: OscillatorField, gen: torch.Generator) -> OscillatorField:
    """Post-hoc control: permute each channel's kernel entries in place —
    destroys learned spatial structure, keeps the magnitude distribution."""
    with torch.no_grad():
        k = physics_block(model.core).kernel
        g2 = k.shape[-1] * k.shape[-2]
        for ch in range(k.shape[0]):
            perm = torch.randperm(g2, generator=gen)
            k[ch] = k[ch].flatten()[perm].view_as(k[ch])
    return model

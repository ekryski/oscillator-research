"""Field instrumentation: mechanism, never verdicts.

Drive-phase locking (PLV against the analytic phase of each row's own
delivered drive), entrained fractions, velocity maps, the global order
parameter R, and kernel spectra. Eval-only; the probes never see any of it,
and no decision bar is ever scored from an instrument.
"""

from __future__ import annotations

import math

import torch

from harness.models import OscillatorField, RandGraphCore, physics_block
from harness.utils.constants import PLV_LOCK_THRESH, TWO_PI, WARMUP_FRAMES


def analytic_row_phase(rows: torch.Tensor) -> torch.Tensor:
    """[B,T,G] band-signal rows -> [B,T,G] instantaneous (analytic) phase.

    FFT-based Hilbert transform per band: the drive's own phase, usable as
    the locking reference when no analytic stimulus phase exists (speech).
     instrument fix: the legacy digits path passed phases==zeros, so
    PLV measured phase CONSTANCY (a rotating-but-entrained oscillator read
    ~0, ent_frac pinned at 0.000). Against the row's carrier phase, PLV
    measures genuine stimulus locking per band."""
    b, t, g = rows.shape
    x = rows.transpose(1, 2).reshape(b * g, t)
    xf = torch.fft.fft(x, dim=1)
    h = torch.zeros(t, dtype=xf.dtype, device=xf.device)
    h[0] = 1.0
    if t % 2 == 0:
        h[t // 2] = 1.0
        h[1:t // 2] = 2.0
    else:
        h[1:(t + 1) // 2] = 2.0
    z = torch.fft.ifft(xf * h, dim=1)  # analytic signal
    return torch.atan2(z.imag, z.real).reshape(b, g, t).transpose(1, 2).contiguous()


@torch.no_grad()
def instrument_field(model: OscillatorField, rows: torch.Tensor, phases: torch.Tensor,
                     labels: torch.Tensor, n_classes: int,
                     warmup: int = WARMUP_FRAMES, batch: int = 64,
                     phi_rows: torch.Tensor | None = None) -> dict:
    """Field diagnostics vs the stimulus: per-class PLV maps, entrained
    fractions, velocity maps, order parameter, kernel spectra.

    phi_rows [B,T,G] (optional,): per-band drive phase used as the
    locking reference instead of the global `phases` — row r's oscillators
    are compared against band r's phase (tonotopy broadcast, mirroring the
    drive path). Required for meaningful PLV/ent_frac on tasks with no
    analytic stimulus phase (digits; any carrier-driven speech)."""
    device = next(model.parameters()).device
    rows = rows.to(device)
    plv_clips, vel_clips, r_means = [], [], []
    for i in range(0, rows.shape[0], batch):
        theta = model.phase_trajectory(rows[i:i + batch])[:, warmup:]  # [b,t,C,G,G]
        if phi_rows is not None:
            # [b,t,G] -> broadcast band r onto grid ROW r, all channels/cols
            phi = phi_rows[i:i + batch, warmup:].to(device)[:, :, None, :, None]
        else:
            phi = phases[i:i + batch, warmup:, None, None, None].to(device)
        diff = theta - phi
        plv = torch.complex(torch.cos(diff).mean(dim=1), torch.sin(diff).mean(dim=1)).abs()
        plv_clips.append(plv)  # [b,C,G,G]
        vel_clips.append(torch.cos(theta[:, 1:] - theta[:, :-1]).mean(dim=1))
        z = torch.complex(torch.cos(theta), torch.sin(theta))
        r_means.append(z.mean(dim=(-2, -1)).abs().mean(dim=(1, 2)))  # [b]
    plv = torch.cat(plv_clips).cpu()
    vel = torch.cat(vel_clips).cpu()
    r_mean = torch.cat(r_means).cpu()
    plv_map = torch.stack([plv[labels == k].mean(dim=0) for k in range(n_classes)])
    vel_map = torch.stack([vel[labels == k].mean(dim=0) for k in range(n_classes)])
    ent_frac = torch.stack([(plv[labels == k] > PLV_LOCK_THRESH).float().mean()
                            for k in range(n_classes)])
    blk = physics_block(model.core) if not isinstance(model.core, RandGraphCore) else model.core
    if not hasattr(blk, "kernel"):
        # randgraph core: no circulant kernel -> spectral stats read the graph
        return dict(plv_map=plv_map, vel_map=vel_map, ent_frac=ent_frac,
                    plv_mean=plv.mean().item(), R_mean=r_mean.mean().item(),
                    ent_frac_mean=ent_frac.mean().item(),
                    omega=blk.natural_freqs.detach().cpu().clone(),
                    opnorm_raw=torch.zeros(1), opnorm_eff=torch.zeros(1),
                    spec_entropy=torch.zeros(1))
    kernel = blk.kernel.detach()
    khat = torch.fft.rfft2(kernel).abs()
    opnorm_raw = khat.flatten(1).max(dim=1).values
    # full-FFT spectrum for entropy (matches the diagnostics-script convention)
    p = torch.fft.fft2(kernel).abs().flatten(1)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log2()).sum(dim=1)
    clamp = blk.spectral_clamp
    opnorm_eff = opnorm_raw.clamp(max=clamp) if clamp else opnorm_raw
    return dict(
        plv_map=plv_map, vel_map=vel_map, ent_frac=ent_frac,
        plv_mean=plv.mean().item(), R_mean=r_mean.mean().item(),
        ent_frac_mean=ent_frac.mean().item(),
        omega=blk.natural_freqs.detach().cpu().clone(),
        opnorm_raw=opnorm_raw.cpu(), opnorm_eff=opnorm_eff.cpu(), spec_entropy=entropy.cpu(),
    )


def natural_rate(omega_mean: float = 1.0, damping: float = 0.5, dt: float = 0.1,
                 substeps: int = 1) -> float:
    """Mean free-rotation rate of the untrained field, cycles/frame
    (tilted-washboard velocity; 0 if pinned, |omega| <= damping)."""
    if abs(omega_mean) <= damping:
        return 0.0
    return substeps * dt * math.sqrt(omega_mean ** 2 - damping ** 2) / TWO_PI

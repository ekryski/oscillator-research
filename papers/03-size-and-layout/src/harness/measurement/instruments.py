"""The oscillator network's instruments: mechanism, never results.

Carried from paper 02's harness (its `analytic_row_phase`). The instruments
themselves (the order parameter, locking to the drive, the entrained share and
the amplitude) are computed per clip by `harness.experiment.arms.field_instruments`
and summarized per run; nothing here reaches the readout.
"""

from __future__ import annotations

import torch


def analytic_row_phase(rows: torch.Tensor) -> torch.Tensor:
    """[B,T,G] band-signal rows -> [B,T,G] instantaneous (analytic) phase.

    FFT-based Hilbert transform per band: the drive's own phase, the locking
    reference when the input carries no phase of its own (the band energies,
    and the carrier's band waveforms)."""
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

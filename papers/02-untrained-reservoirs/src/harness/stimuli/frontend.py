"""Hop-frame frontend: fixed log-mel rows at 62.5 fps.

The envelope rung of the transduction ladder. Contract difference vs
`bandpass_rows`, recorded: rows are log-energy ENVELOPE trajectories per mel
band — nonnegative, slow (62.5 fps), tonotopic by construction (mel band b ->
row b, ascending frequency) — not sample-rate waveforms. The field is driven
by spectral envelopes, the classic reservoir input. Deliberately FIXED (no
trained parameters, no per-utterance statistics — the streaming-honesty rule)
so the bookends cannot dominate the measurement; the whole map is
deterministic.

A ~1 s digit gives ~60 frames, so whole-word scans are cheap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.stimuli.audio import MelFrontend

HOP_N_FFT = 512
HOP_LENGTH = 256          # @16 kHz -> 62.5 fps
HOP_N_ROWS = 16           # matches the harness grid contract (band b -> row b)
# Fixed affine from log-mel (~[-13, +5] on unit-scale audio) into a bounded,
# mostly-nonnegative O(1) drive range: silence -> ~0, loud bands -> ~1.5.
HOP_OFFSET = 10.0
HOP_SCALE = 10.0

_MELS: dict[tuple[int, int], MelFrontend] = {}


def hop_num_frames(n_samples: int) -> int:
    """Exact frame count (center=False): T = (L - n_fft) // hop + 1."""
    return max(0, (n_samples - HOP_N_FFT) // HOP_LENGTH + 1)


@torch.no_grad()
def hop_rows(waves: torch.Tensor, grid: int = HOP_N_ROWS,
             sample_rate: int = 16000) -> torch.Tensor:
    """waves [B, L] -> rows [B, T, grid], deterministic fixed map.

    log-mel via the pipeline's MelFrontend (log(mel + eps)), then the fixed
    affine (x + HOP_OFFSET) / HOP_SCALE clamped at 0 — no learned or
    per-utterance quantities anywhere."""
    key = (grid, sample_rate)
    if key not in _MELS:
        _MELS[key] = MelFrontend(sample_rate=sample_rate, n_fft=HOP_N_FFT,
                                 hop=HOP_LENGTH, n_mels=grid)
    logmel = _MELS[key](waves)                       # [B, T, grid]
    return torch.clamp((logmel + HOP_OFFSET) / HOP_SCALE, min=0.0)


# ---------------------------------------------------------------------------
# Quadrature-baseband variant (phase factor; gate build 1b)
# ---------------------------------------------------------------------------
# Per band, the STFT bin nearest the band's weighted center supplies the
# band's complex analytic sample; DEMODULATING by that center frequency
# (phi_baseband = phi - 2*pi*f_c*t) leaves a slowly-varying phase that is
# valid at hop rate. Physics-honest scoping (a Nyquist theorem, stated in
# the papers): hop-rate phase can only ever be BASEBAND deviation within a
# band (±fps/2 = ±31.25 Hz), never raw carrier cycles — the sample-rate
# frontend remains the carrier-true instrument.

_QUAD: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _quad_maps(grid: int, sample_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(band -> chosen STFT bin index [grid], bin center freq Hz [grid]),
    from the same mel filterbank the magnitude path uses (band centers =
    each filter's peak bin) — one tonotopy for both frontends."""
    key = (grid, sample_rate)
    if key not in _QUAD:
        mkey = (grid, sample_rate)
        if mkey not in _MELS:
            _MELS[mkey] = MelFrontend(sample_rate=sample_rate, n_fft=HOP_N_FFT,
                                      hop=HOP_LENGTH, n_mels=grid)
        fb = _MELS[mkey].mel.mel_scale.fb              # [n_freqs, n_mels]
        bins = fb.argmax(dim=0)                        # [grid] peak bin per band
        freqs = bins.to(torch.float32) * sample_rate / HOP_N_FFT
        _QUAD[key] = (bins, freqs)
    return _QUAD[key]


@torch.no_grad()
def hop_rows_quad(waves: torch.Tensor, grid: int = HOP_N_ROWS,
                  sample_rate: int = 16000) -> torch.Tensor:
    """waves [B, L] -> [B, T, grid, 2]: per band (A·cos φ_bb, A·sin φ_bb).

    A = the magnitude path's row value (identical envelope in both
    frontends — the factor isolates PHASE); φ_bb = the band-center STFT
    bin's phase, demodulated by the bin frequency. Deterministic, fixed."""
    amp = hop_rows(waves, grid, sample_rate)           # [B, T, grid]
    bins, freqs = _quad_maps(grid, sample_rate)
    window = torch.hann_window(HOP_N_FFT)
    spec = torch.stft(waves, n_fft=HOP_N_FFT, hop_length=HOP_LENGTH,
                      window=window, center=False, return_complex=True)
    spec = spec[:, bins, :].transpose(1, 2)            # [B, T, grid] complex
    t = torch.arange(spec.shape[1], dtype=torch.float32)[None, :, None]
    demod = torch.exp(-2j * torch.pi * freqs[None, None, :]
                      * (t * HOP_LENGTH / sample_rate))
    phi = torch.angle(spec * demod)                    # baseband phase [B,T,grid]
    return torch.stack((amp * torch.cos(phi), amp * torch.sin(phi)), dim=-1)

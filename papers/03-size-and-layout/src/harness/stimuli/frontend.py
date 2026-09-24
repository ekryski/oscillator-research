"""Hop-frame front ends: fixed log-mel rows at 62.5 frames per second.

The band-energy pathway: rows are log-energy trajectories per mel band,
non-negative, slow and in frequency order (mel band b drives row b). Fixed:
no trained parameters and no per-utterance statistics. Every band count uses
paper 02's frames: a 512-sample (32 ms) Hann window every 256 samples.

Band counts above 32. A 512-point transform has 257 frequency bins, and the
narrowest mel filters at 64 and 128 bands fall between them: at 128 bands one
filter is empty and 25 touch a single bin. From 32 bands up the transform is
therefore zero-padded so the narrowest filter always spans at least three
bins, as the 32-band filter does at 512 points: 1,024 points at 64 bands and
2,048 at 128. The window is still 512 samples, shifted back onto paper 02's
frames, so the frame count and timing are unchanged; zero-padding samples the
same windowed spectrum more finely and adds no frequency resolution, so
adjacent low bands at 64 and 128 are strongly correlated. At 32 bands and
below nothing changes: 8, 16 and 32 bands are computed exactly as paper 02
computed them.

A digit of about 1 s gives about 60 frames.
"""
from __future__ import annotations

import torch

from harness.stimuli.audio import MelFrontend

HOP_N_FFT = 512           # the analysis window, in samples, at every band count
HOP_LENGTH = 256          # @16 kHz -> 62.5 fps
HOP_N_ROWS = 16           # paper 02's band count
#: above this many bands the transform is zero-padded (see the module docstring)
UNPADDED_BANDS = 32
# Fixed affine from log-mel (~[-13, +5] on unit-scale audio) into a bounded,
# mostly-nonnegative O(1) drive range: silence -> ~0, loud bands -> ~1.5.
HOP_OFFSET = 10.0
HOP_SCALE = 10.0

_MELS: dict[tuple[int, int], MelFrontend] = {}


def fft_points(bands: int) -> int:
    """Transform length for a band count: 512 up to 32 bands, then growing with the band count."""
    return HOP_N_FFT if bands <= UNPADDED_BANDS else HOP_N_FFT * bands // UNPADDED_BANDS


def hop_num_frames(n_samples: int) -> int:
    """Exact frame count (center=False, paper 02's frames): T = (L - 512) // 256 + 1."""
    return max(0, (n_samples - HOP_N_FFT) // HOP_LENGTH + 1)


def _mel(grid: int, sample_rate: int) -> MelFrontend:
    key = (grid, sample_rate)
    if key not in _MELS:
        n_fft = fft_points(grid)
        if n_fft == HOP_N_FFT:
            _MELS[key] = MelFrontend(sample_rate=sample_rate, n_fft=HOP_N_FFT, hop=HOP_LENGTH, n_mels=grid)
        else:
            _MELS[key] = MelFrontend(sample_rate=sample_rate, n_fft=n_fft, hop=HOP_LENGTH, n_mels=grid,
                                     win_length=HOP_N_FFT, pad=(n_fft - HOP_N_FFT) // 2)
    return _MELS[key]


@torch.no_grad()
def hop_rows(waves: torch.Tensor, grid: int = HOP_N_ROWS, sample_rate: int = 16000) -> torch.Tensor:
    """waves [B, L] -> rows [B, T, grid]: log-mel, then the fixed affine (x + 10) / 10 clamped at 0."""
    logmel = _mel(grid, sample_rate)(waves)          # [B, T, grid]
    return torch.clamp((logmel + HOP_OFFSET) / HOP_SCALE, min=0.0)


# ---------------------------------------------------------------------------
# The quadrature pathway
# ---------------------------------------------------------------------------
# Per band, the transform bin at the peak of the band's mel filter supplies a
# complex sample; demodulating by that bin's frequency leaves a slowly varying
# baseband phase that is valid at the hop rate (within +-31.25 Hz of the bin).

_QUAD: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _quad_maps(grid: int, sample_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(band -> peak bin index [grid], that bin's frequency in Hz [grid]), from the band-energy filters."""
    key = (grid, sample_rate)
    if key not in _QUAD:
        fb = _mel(grid, sample_rate).mel.mel_scale.fb   # [n_freqs, n_mels]
        bins = fb.argmax(dim=0)                        # [grid] peak bin per band
        freqs = bins.to(torch.float32) * sample_rate / fft_points(grid)
        _QUAD[key] = (bins, freqs)
    return _QUAD[key]


@torch.no_grad()
def hop_rows_quad(waves: torch.Tensor, grid: int = HOP_N_ROWS, sample_rate: int = 16000) -> torch.Tensor:
    """waves [B, L] -> [B, T, grid, 2]: per band (A cos phi_bb, A sin phi_bb).

    A is the band-energy row (the same envelope as the band-energy pathway, so
    the pathway differs in phase alone); phi_bb is the peak bin's phase,
    demodulated by the bin frequency and referenced to the start of paper
    02's 512-sample window, so a zero-padded transform gives the same phase
    as the unpadded one wherever their bins coincide."""
    amp = hop_rows(waves, grid, sample_rate)           # [B, T, grid]
    bins, freqs = _quad_maps(grid, sample_rate)
    n_fft = fft_points(grid)
    if n_fft == HOP_N_FFT:                             # paper 02's arithmetic, unchanged
        spec = torch.stft(waves, n_fft=HOP_N_FFT, hop_length=HOP_LENGTH, window=torch.hann_window(HOP_N_FFT),
                          center=False, return_complex=True)
        spec = spec[:, bins, :].transpose(1, 2)        # [B, T, grid] complex
        t = torch.arange(spec.shape[1], dtype=torch.float32)[None, :, None]
        when = t * HOP_LENGTH / sample_rate
    else:
        pad = (n_fft - HOP_N_FFT) // 2
        spec = torch.stft(torch.nn.functional.pad(waves, (pad, pad)), n_fft=n_fft, hop_length=HOP_LENGTH,
                          win_length=HOP_N_FFT, window=torch.hann_window(HOP_N_FFT), center=False,
                          return_complex=True)
        spec = spec[:, bins, :].transpose(1, 2)
        t = torch.arange(spec.shape[1], dtype=torch.float32)[None, :, None]
        when = (t * HOP_LENGTH - pad) / sample_rate    # each transform frame starts `pad` before its window
    demod = torch.exp(-2j * torch.pi * freqs[None, None, :] * when)
    phi = torch.angle(spec * demod)                    # baseband phase [B, T, grid]
    return torch.stack((amp * torch.cos(phi), amp * torch.sin(phi)), dim=-1)

"""Hop-frame front ends: fixed log-mel rows at 62.5 frames per second.

The spectrogram pathway: rows are log-energy trajectories per mel band,
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

The window as a front-end variable. The analysis window can also grow with
the band count, `window` samples long (1,024 at 64 bands, 2,048 at 128 in the
design), which resolves the low bands the 512-sample window cannot, at the
cost of time resolution: each frame then averages 64 or 128 ms of audio
instead of 32. Its transform is `window` points (or more, if the band count
needs more), its frames are centred where paper 02's are, and a clip has the
same number of frames, so the read is unchanged. At the default window of 512
samples every band count is computed as above.

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

_MELS: dict[tuple[int, int, int, str], MelFrontend] = {}


def fft_points(bands: int, window: int = HOP_N_FFT) -> int:
    """Transform length: 512 up to 32 bands, then growing with the band count; at least the window."""
    return max(window, HOP_N_FFT if bands <= UNPADDED_BANDS else HOP_N_FFT * bands // UNPADDED_BANDS)


def frame_pad(bands: int, window: int = HOP_N_FFT) -> int:
    """Zeros on each side of the waveform that centre every transform frame where paper 02's frame is:
    frame k's centre is sample 256 k + 256 whatever the transform's length."""
    return (fft_points(bands, window) - HOP_N_FFT) // 2


def hop_num_frames(n_samples: int) -> int:
    """Exact frame count (center=False, paper 02's frames): T = (L - 512) // 256 + 1."""
    return max(0, (n_samples - HOP_N_FFT) // HOP_LENGTH + 1)


def _check_window(window: int) -> None:
    if window < HOP_N_FFT or window % HOP_LENGTH:
        raise ValueError(f"a window of {window} samples: it must be at least {HOP_N_FFT} and a multiple of "
                         f"the {HOP_LENGTH}-sample hop")


def _mel(grid: int, sample_rate: int, device: torch.device | str = "cpu",
         window: int = HOP_N_FFT) -> MelFrontend:
    """The mel front end for a band count and window, on its waveforms' device (built once per device)."""
    _check_window(window)
    key = (grid, sample_rate, window, str(device))
    if key not in _MELS:
        n_fft = fft_points(grid, window)
        if n_fft == HOP_N_FFT:                        # paper 02's front end, unchanged
            mel = MelFrontend(sample_rate=sample_rate, n_fft=HOP_N_FFT, hop=HOP_LENGTH, n_mels=grid)
        else:
            mel = MelFrontend(sample_rate=sample_rate, n_fft=n_fft, hop=HOP_LENGTH, n_mels=grid,
                              win_length=window, pad=frame_pad(grid, window))
        _MELS[key] = mel.to(device)
    return _MELS[key]


@torch.no_grad()
def hop_rows(waves: torch.Tensor, grid: int = HOP_N_ROWS, sample_rate: int = 16000,
             window: int = HOP_N_FFT) -> torch.Tensor:
    """waves [B, L] -> rows [B, T, grid]: log-mel, then the fixed affine (x + 10) / 10 clamped at 0."""
    logmel = _mel(grid, sample_rate, waves.device, window)(waves)          # [B, T, grid]
    return torch.clamp((logmel + HOP_OFFSET) / HOP_SCALE, min=0.0)


# ---------------------------------------------------------------------------
# The quadrature pathway
# ---------------------------------------------------------------------------
# Per band, the transform bin at the peak of the band's mel filter supplies a
# complex sample; demodulating by that bin's frequency leaves a slowly varying
# baseband phase that is valid at the hop rate (within +-31.25 Hz of the bin).

_QUAD: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _quad_maps(grid: int, sample_rate: int, window: int = HOP_N_FFT) -> tuple[torch.Tensor, torch.Tensor]:
    """(band -> peak bin index [grid], that bin's frequency in Hz [grid]), from the mel filters."""
    key = (grid, sample_rate, window)
    if key not in _QUAD:
        fb = _mel(grid, sample_rate, window=window).mel.mel_scale.fb   # [n_freqs, n_mels], the CPU copy
        bins = fb.argmax(dim=0)                        # [grid] peak bin per band
        freqs = bins.to(torch.float32) * sample_rate / fft_points(grid, window)
        _QUAD[key] = (bins, freqs)
    return _QUAD[key]


@torch.no_grad()
def hop_rows_quad(waves: torch.Tensor, grid: int = HOP_N_ROWS, sample_rate: int = 16000,
                  window: int = HOP_N_FFT) -> torch.Tensor:
    """waves [B, L] -> [B, T, grid, 2]: per band (A cos phi_bb, A sin phi_bb).

    A is the band energy row (the same envelope as the spectrogram pathway, so
    the pathway differs in phase alone); phi_bb is the peak bin's phase,
    demodulated by the bin frequency and referenced to the start of each
    transform frame, which for a zero-padded transform gives paper 02's phase
    wherever the bins coincide."""
    amp = hop_rows(waves, grid, sample_rate, window)   # [B, T, grid]
    bins, freqs = _quad_maps(grid, sample_rate, window)
    bins, freqs = bins.to(waves.device), freqs.to(waves.device)
    n_fft = fft_points(grid, window)
    if n_fft == HOP_N_FFT:                             # paper 02's arithmetic, unchanged
        hann = torch.hann_window(HOP_N_FFT, device=waves.device)
        spec = torch.stft(waves, n_fft=HOP_N_FFT, hop_length=HOP_LENGTH, window=hann, center=False,
                          return_complex=True)
        spec = spec[:, bins, :].transpose(1, 2)        # [B, T, grid] complex
        t = torch.arange(spec.shape[1], dtype=torch.float32, device=waves.device)[None, :, None]
        when = t * HOP_LENGTH / sample_rate
    else:
        pad = frame_pad(grid, window)
        spec = torch.stft(torch.nn.functional.pad(waves, (pad, pad)), n_fft=n_fft, hop_length=HOP_LENGTH,
                          win_length=window, window=torch.hann_window(window, device=waves.device),
                          center=False, return_complex=True)
        spec = spec[:, bins, :].transpose(1, 2)
        t = torch.arange(spec.shape[1], dtype=torch.float32, device=waves.device)[None, :, None]
        when = (t * HOP_LENGTH - pad) / sample_rate    # each transform frame starts `pad` before paper 02's
    demod = torch.exp(-2j * torch.pi * freqs[None, None, :] * when)
    phi = torch.angle(spec * demod)                    # baseband phase [B, T, grid]
    return torch.stack((amp * torch.cos(phi), amp * torch.sin(phi)), dim=-1)

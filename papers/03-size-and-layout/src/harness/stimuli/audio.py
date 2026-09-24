"""The log-mel front end behind the band-energy pathway."""

from __future__ import annotations

import torch
import torchaudio
from torch import nn

EPS = 1e-5


class MelFrontend(nn.Module):
    """Log-mel filterbank energies, with nothing trained or fitted.

    center=False so the frame count is exact: T = (L + 2 pad - n_fft) // hop + 1.
    No per-utterance normalization: nothing depends on having seen the whole clip.

    `win_length` shorter than `n_fft` zero-pads each windowed frame to n_fft
    points, which samples the same spectrum on a finer frequency grid; `pad`
    zeros on each side of the waveform put that window back on paper 02's
    frames. Left at their defaults the transform is paper 02's exactly.
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024, hop: int = 256,
                 n_mels: int = 80, win_length: int | None = None, pad: int = 0):
        super().__init__()
        self.n_fft, self.hop, self.n_mels, self.pad = n_fft, hop, n_mels, pad
        extra = {} if win_length is None else {"win_length": win_length, "pad": pad}
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            center=False,
            power=2.0,
            **extra,
        )

    def num_frames(self, lens: torch.Tensor) -> torch.Tensor:
        return torch.clamp((lens + 2 * self.pad - self.n_fft) // self.hop + 1, min=0)

    def forward(self, waves: torch.Tensor) -> torch.Tensor:
        """waves [B, L] -> raw log-mel [B, T, n_mels]."""
        return torch.log(self.mel(waves) + EPS).transpose(1, 2)  # [B, T, M]

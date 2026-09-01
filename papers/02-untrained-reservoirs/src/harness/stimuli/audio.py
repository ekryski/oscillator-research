"""The log-mel front-end behind the envelope drive pathway."""

from __future__ import annotations

import torch
import torchaudio
from torch import nn

EPS = 1e-5


class MelFrontend(nn.Module):
    """Log-mel filterbank features, with zero trainable parameters.

    center=False so the frame count is exact: T = (L - n_fft) // hop + 1.

    Deliberately NO per-utterance normalization: utterance-level statistics are
    unknowable mid-stream, so nothing here depends on having seen the whole
    clip. The fixed affine that turns these log-mels into a bounded drive range
    lives in `harness.frontend` — also parameter-free.
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024, hop: int = 256,
                 n_mels: int = 80):
        super().__init__()
        self.n_fft, self.hop, self.n_mels = n_fft, hop, n_mels
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            center=False,
            power=2.0,
        )

    def num_frames(self, lens: torch.Tensor) -> torch.Tensor:
        return torch.clamp((lens - self.n_fft) // self.hop + 1, min=0)

    def forward(self, waves: torch.Tensor, lens: torch.Tensor | None = None) -> torch.Tensor:
        """waves [B, L] -> raw log-mel [B, T, n_mels]; padded frames zeroed when lens given."""
        feats = torch.log(self.mel(waves) + EPS).transpose(1, 2)  # [B, T, M]
        if lens is None:
            return feats
        t = feats.size(1)
        mask3 = (torch.arange(t, device=feats.device)[None, :]
                 < self.num_frames(lens)[:, None])[:, :, None]
        return feats * mask3

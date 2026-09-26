"""The phase summary the oscillator network's own feature methods use, and the pooling the trained baselines use."""

from __future__ import annotations

import torch

from harness.utils.constants import WARMUP_FRAMES


def phase_features(feats: torch.Tensor, channels: int, grid: int,
                   warmup: int = WARMUP_FRAMES,
                   tvalid: torch.Tensor | None = None) -> torch.Tensor:
    """[B,T,2*C*G*G] scan features -> [B, 4*C*G*G] clip features.

    Per oscillator: time-mean sin/cos (position signature) and time-mean
    cos/sin of the per-frame phase INCREMENT (velocity signature — a locked
    oscillator's increment encodes the stimulus frequency; an unlocked one's
    encodes its own omega). Whole-clip means, no per-segment windows: temporal
    order information can only enter through dynamical (hysteresis) effects."""
    d = channels * grid * grid
    s, c = feats[..., :d], feats[..., d:]
    if tvalid is None:
        s_m = s[:, warmup:].mean(dim=1)
        c_m = c[:, warmup:].mean(dim=1)
        s1, s0 = s[:, warmup:], s[:, warmup - 1:-1]
        c1, c0 = c[:, warmup:], c[:, warmup - 1:-1]
        cd = (c1 * c0 + s1 * s0).mean(dim=1)  # cos(theta_t - theta_{t-1})
        sd = (s1 * c0 - c1 * s0).mean(dim=1)  # sin(theta_t - theta_{t-1})
        return torch.cat((s_m, c_m, cd, sd), dim=1)
    # length masking: pool frames [warmup, tvalid_i) per clip — padded
    # ring-down never enters the statistics (same contract, masked means).
    t = feats.shape[1]
    m = ((torch.arange(t, device=feats.device)[None, :] >= warmup)
         & (torch.arange(t, device=feats.device)[None, :] < tvalid.to(feats.device)[:, None]))
    mf = m[:, :, None].float()
    n = mf.sum(dim=1).clamp_min(1.0)
    s_m = (s * mf).sum(dim=1) / n
    c_m = (c * mf).sum(dim=1) / n
    s1, s0 = s[:, 1:], s[:, :-1]
    c1, c0 = c[:, 1:], c[:, :-1]
    md = mf[:, 1:]  # increment valid when frame t valid (t-1 then also < tvalid)
    nd = md.sum(dim=1).clamp_min(1.0)
    cd = ((c1 * c0 + s1 * s0) * md).sum(dim=1) / nd
    sd = ((s1 * c0 - c1 * s0) * md).sum(dim=1) / nd
    return torch.cat((s_m, c_m, cd, sd), dim=1)


def pooled_stats(h: torch.Tensor, tvalid: torch.Tensor | None = None) -> torch.Tensor:
    """Shared hidden-trajectory pooling for the conventional references:
    (mean, std, |delta| mean, last) over time — the GRU/TCN feature contract.
    tvalid: pool valid frames only; "last" = the last VALID frame."""
    if tvalid is None:
        dh = (h[:, 1:] - h[:, :-1]).abs().mean(dim=1)
        return torch.cat((h.mean(dim=1), h.std(dim=1), dh, h[:, -1]), dim=1)
    t = h.shape[1]
    tv = tvalid.to(h.device).clamp(2, t)
    m = (torch.arange(t, device=h.device)[None, :] < tv[:, None])[:, :, None].float()
    n = m.sum(dim=1)
    mean = (h * m).sum(dim=1) / n
    var = ((h - mean[:, None]) ** 2 * m).sum(dim=1) / (n - 1).clamp_min(1.0)
    md = m[:, 1:]
    dh = ((h[:, 1:] - h[:, :-1]).abs() * md).sum(dim=1) / md.sum(dim=1).clamp_min(1.0)
    last = h[torch.arange(h.shape[0], device=h.device), tv - 1]
    # sqrt(var + eps), masked branch only: a dead (constant) unit has EXACTLY
    # zero variance over a short valid window (measured: ReLU nets on digits),
    # and sqrt'(0) = inf turns its zero upstream grad into NaN (inf * 0),
    # killing training. The eps keeps gradients finite; the value shift
    # (<= 1e-4) is far below hidden-state scale. The unmasked branch above is
    # untouched — frozen controls depend on its exact numerics.
    return torch.cat((mean, (var + 1e-8).sqrt(), dh, last), dim=1)


def windowed_stats(h: torch.Tensor, windows: int,
                    tvalid: torch.Tensor | None = None) -> torch.Tensor:
    """Windowed-parity pooling (item 7): same windowing contract as
    OscillatorField.features_windowed so ridge comparisons stay arm-fair. Each
    window needs >= 2 frames for std/delta to be defined.

    tvalid (length masking, in h's ALREADY-WARMUP-CROPPED frame
    coordinates): per-clip windows over [0, tvalid_i); clips too short for
    2-frame windows extend into the trailing frames (hi >= 2*windows, clamped
    to T) — the same rule as OscillatorField.features_windowed; tvalid == T
    reproduces the unmasked windows exactly. Per-clip loop, eval-only."""
    t = h.shape[1]
    if tvalid is None:
        assert t // windows >= 2, f"window too short: {t} frames / {windows} windows"
        return torch.cat([pooled_stats(h[:, i * t // windows:(i + 1) * t // windows])
                          for i in range(windows)], dim=1)
    out = []
    for i in range(h.shape[0]):
        hi = min(t, max(int(tvalid[i]), 2 * windows))
        out.append(torch.cat([pooled_stats(h[i:i + 1, hi * j // windows: hi * (j + 1) // windows])
                              for j in range(windows)], dim=1))
    return torch.cat(out, dim=0)


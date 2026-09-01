"""Synthetic stimuli with analytically known structure.

These are the contract stimuli, not the paper's task. Their value is that the
right answer is known in closed form: a tone's energy belongs in one band, and
the two frequency-step orders have provably identical average spectra, so the
tests can pin what the field and the readout are each allowed to see. The step
task in particular is the template for the paper's order task — statics
provably at chance, so anything above it is memory.
"""

from __future__ import annotations

import math

import torch

from harness.stimuli.filterbank import band_edges
from harness.utils.constants import TWO_PI


def tone_classes(grid: int, rows: tuple[int, ...] = (3, 6, 9, 12),
                 positions: tuple[float, ...] = (0.35, 0.65)) -> list[tuple[float, int]]:
    """Tone discrimination classes as (freq, row) pairs.

    Two tones per band (geometric positions within it): the spatial cue "which
    row got energy" is shared within each pair, so separating pair members
    requires the field's temporal response — this is the hardening against a
    static energy-map readout solving the task without any dynamics."""
    e = band_edges(grid)
    out = []
    for r in rows:
        lo, hi = float(e[r]), float(e[r + 1])
        for p in positions:
            out.append((lo * (hi / lo) ** p, r))
    return out


def make_tone_clips(n: int, frames: int, classes: list[tuple[float, int]],
                    gen: torch.Generator, noise_db: float | None = -20.0,
                    freq_jitter: float = 0.015) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-tone clips -> (wave [n,T] unit-RMS, stim_phase [n,T], labels [n]).

    Nuisance parameters per clip (random initial phase, +-freq_jitter, white
    noise) make memorization impossible; unit-RMS normalization removes energy
    as a cue entirely. stim_phase is the true instantaneous phase, recorded for
    PLV instrumentation only — the probes never see it."""
    k = len(classes)
    labels = (torch.arange(n) % k)[torch.randperm(n, generator=gen)]
    freqs = torch.tensor([c[0] for c in classes])[labels]
    freqs = freqs * (1 + (torch.rand(n, generator=gen) * 2 - 1) * freq_jitter)
    phi0 = torch.rand(n, generator=gen) * TWO_PI
    t = torch.arange(frames, dtype=torch.float32)
    phase = TWO_PI * freqs[:, None] * t[None, :] + phi0[:, None]
    wave = torch.sin(phase)
    if noise_db is not None:
        sigma = (1 / math.sqrt(2)) * 10 ** (noise_db / 20)  # relative to tone RMS
        wave = wave + sigma * torch.randn(n, frames, generator=gen)
    wave = wave / wave.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    return wave, phase, labels


def step_freqs(grid: int, rows: tuple[int, int] = (5, 10)) -> tuple[float, float]:
    """The two band-center frequencies used by the step-order task."""
    e = band_edges(grid)
    return tuple(float(math.sqrt(e[r] * e[r + 1])) for r in rows)  # type: ignore[return-value]


def make_step_clips(n: int, frames: int, f_a: float, f_b: float,
                    gen: torch.Generator, noise_db: float | None = -20.0,
                    switch_jitter: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frequency-step clips: label 0 = A->B, label 1 = B->A (phase-continuous).

    Both classes have identical average spectra (half the clip at each
    frequency, symmetric switch-time distribution), so a readout of whole-clip
    mean features can separate them ONLY through temporal-order effects in the
    dynamics — this is the task where statics provably sit at chance."""
    labels = (torch.arange(n) % 2)[torch.randperm(n, generator=gen)]
    t = torch.arange(frames, dtype=torch.float32)
    ts = frames // 2 + (torch.randint(-switch_jitter, switch_jitter + 1, (n,), generator=gen)
                        if switch_jitter else torch.zeros(n, dtype=torch.long))
    f1 = torch.where(labels == 0, torch.tensor(f_a), torch.tensor(f_b))
    f2 = torch.where(labels == 0, torch.tensor(f_b), torch.tensor(f_a))
    phi0 = torch.rand(n, generator=gen) * TWO_PI
    # phase-continuous switch: phi(t) = phi0 + 2*pi*f1*min(t,ts) + 2*pi*f2*max(t-ts,0)
    seg1 = torch.minimum(t[None, :], ts[:, None].float())
    seg2 = (t[None, :] - ts[:, None].float()).clamp_min(0)
    phase = phi0[:, None] + TWO_PI * (f1[:, None] * seg1 + f2[:, None] * seg2)
    wave = torch.sin(phase)
    if noise_db is not None:
        sigma = (1 / math.sqrt(2)) * 10 ** (noise_db / 20)
        wave = wave + sigma * torch.randn(n, frames, generator=gen)
    wave = wave / wave.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    return wave, phase, labels

# ---------------------------------------------------------------------------

AM_CARRIER_ROW = 6  # carrier sits at the band-6 center for every class


def am_rates(n_classes: int = 8, lo: float = 0.002, hi: float = 0.016) -> list[float]:
    """Log-spaced AM (envelope) rates, cycles/frame. Defaults span 3 octaves:
    ~1 to ~8 envelope cycles per 512-frame clip."""
    return [lo * (hi / lo) ** (k / (n_classes - 1)) for k in range(n_classes)]


def make_am_clips(n: int, frames: int, rates: list[float], grid: int,
                  gen: torch.Generator, noise_db: float | None = -20.0,
                  rate_jitter: float = 0.015) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AM-tone clips -> (wave [n,T] unit-RMS, carrier_phase [n,T], labels [n]).

    One shared carrier (band-6 center) for every class; classes differ ONLY in
    envelope rate: wave = (1 + sin(2*pi*fm*t + phi_m))/2 * sin(carrier phase).
    Honest caveat (recorded): AM sidebands at fc +- fm are an inherent spectral
    cue (Fourier duality — different envelope rates cannot have identical
    spectra), so unlike steps this task is NOT statics-provably-at-chance; the
    frozen arm's calibrated ridge accuracy IS the statics+reservoir baseline,
    and verdicts are deltas against it, per protocol. The physics prediction
    under test: envelopes are the washboard DC channel's cargo, so pinning
    level should matter more here than anywhere else."""
    k = len(rates)
    e = band_edges(grid)
    fc = float(math.sqrt(e[AM_CARRIER_ROW] * e[AM_CARRIER_ROW + 1]))
    labels = (torch.arange(n) % k)[torch.randperm(n, generator=gen)]
    fm = torch.tensor(rates)[labels]
    fm = fm * (1 + (torch.rand(n, generator=gen) * 2 - 1) * rate_jitter)
    phi_c = torch.rand(n, generator=gen) * TWO_PI
    phi_m = torch.rand(n, generator=gen) * TWO_PI
    t = torch.arange(frames, dtype=torch.float32)
    carrier_phase = TWO_PI * fc * t[None, :] + phi_c[:, None]
    env = 0.5 * (1 + torch.sin(TWO_PI * fm[:, None] * t[None, :] + phi_m[:, None]))
    wave = env * torch.sin(carrier_phase)
    if noise_db is not None:
        sigma = (1 / math.sqrt(2)) * 10 ** (noise_db / 20)
        wave = wave + sigma * torch.randn(n, frames, generator=gen)
    wave = wave / wave.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    return wave, carrier_phase, labels

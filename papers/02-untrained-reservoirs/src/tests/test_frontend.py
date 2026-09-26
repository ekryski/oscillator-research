"""Contracts for `harness.frontend`.

The frontend is fixed and parameter-free by design, so that the bookend
projections cannot dominate attribution. These tests pin the exact frame
count, the tonotopic band-to-row map, and the invariant that the quadrature
pathway differs from the magnitude pathway in phase alone."""

import math
import sys

import torch

sys.path.insert(0, ".")
from harness.stimuli import PAIR_MAX_SAMPLES
from harness.stimuli.frontend import (
    HOP_LENGTH,
    HOP_N_FFT,
    hop_num_frames,
    hop_rows,
    hop_rows_quad,
)

TWO_PI = 2 * math.pi


def test_hop_frontend_contract():
    # gate build 1: shape math, determinism, tonotopic ordering, silence
    from harness.stimuli.frontend import HOP_LENGTH, HOP_N_FFT, hop_num_frames, hop_rows
    sr, dur = 16000, 16000  # 1 s
    t_expect = (dur - HOP_N_FFT) // HOP_LENGTH + 1
    assert hop_num_frames(dur) == t_expect
    ts = torch.arange(dur, dtype=torch.float32) / sr
    tones = torch.stack([torch.sin(2 * torch.pi * f * ts) for f in (200.0, 1000.0, 4000.0)])
    r1, r2 = hop_rows(tones), hop_rows(tones)
    assert r1.shape == (3, t_expect, 16) and torch.isfinite(r1).all()
    assert torch.equal(r1, r2)  # deterministic
    assert (r1 >= 0).all()
    # ascending tone frequency -> strictly ascending dominant row (tonotopy)
    rows = [r1[i].mean(dim=0).argmax().item() for i in range(3)]
    assert rows[0] < rows[1] < rows[2], rows
    # silence maps to (near-)zero drive
    assert hop_rows(torch.zeros(1, dur)).abs().max().item() < 0.05


def test_hop_quad_frontend_contract():
    # gate 1b: quadrature-baseband rows — magnitude parity with the
    # magnitude path, and the demodulation property (the physics): a tone AT
    # a band center has ~static baseband phase; a tone OFFSET by df advances
    # phase at 2*pi*df per second.
    from harness.stimuli.frontend import HOP_LENGTH, _quad_maps, hop_rows, hop_rows_quad
    sr, dur = 16000, 16000
    bins, freqs = _quad_maps(16, sr)
    fc = float(freqs[8])
    ts = torch.arange(dur, dtype=torch.float32) / sr
    on = torch.sin(2 * torch.pi * fc * ts)[None]
    off = torch.sin(2 * torch.pi * (fc + 10.0) * ts)[None]
    q = hop_rows_quad(torch.cat([on, off]))
    assert q.shape[2:] == (16, 2) and torch.isfinite(q).all()
    assert torch.equal(q, hop_rows_quad(torch.cat([on, off])))  # deterministic
    amp = hop_rows(torch.cat([on, off]))
    assert torch.allclose(q.pow(2).sum(-1).sqrt(), amp, atol=1e-4)  # A preserved
    phi = torch.atan2(q[..., 1], q[..., 0])[:, :, 8]  # band-8 baseband phase
    dphi = torch.remainder(phi[:, 1:] - phi[:, :-1] + torch.pi, 2 * torch.pi) - torch.pi
    v_on, v_off = dphi[0].median().item(), dphi[1].median().item()
    expect_off = 2 * torch.pi * 10.0 * HOP_LENGTH / sr  # rad/frame at +10 Hz
    assert abs(v_on) < 0.05, v_on                      # on-center: ~static
    assert abs(v_off - expect_off) < 0.05, (v_off, expect_off)


def test_hop_frame_count_is_exact():
    for samples in (512, 4352, 16000, PAIR_MAX_SAMPLES):
        assert hop_num_frames(samples) == (samples - HOP_N_FFT) // HOP_LENGTH + 1
    assert hop_num_frames(16000) == 61, "a 1 s clip is 61 hop frames at 62.5 fps"
    assert hop_num_frames(100) == 0, "shorter than one window yields no frames"


def test_hop_rows_are_nonnegative_and_bounded_by_the_fixed_affine():
    """The affine is deliberately fixed and clamped at zero: silence maps to
    ~0 and loud bands to ~1.5, with no per-utterance statistic anywhere."""
    torch.manual_seed(0)
    rows = hop_rows(torch.randn(2, 16000) * 0.2)
    assert rows.shape == (2, 61, 16)
    assert torch.all(rows >= 0.0)
    assert torch.isfinite(rows).all()
    silence = hop_rows(torch.zeros(1, 16000))
    assert float(silence.max()) == 0.0, "digital silence clamps to exactly zero"


def test_quadrature_amplitude_equals_the_magnitude_pathway():
    """The factor isolates PHASE: both pathways carry the identical envelope,
    so any difference between them is attributable to the drive form alone."""
    torch.manual_seed(0)
    waves = torch.randn(2, 16000) * 0.2
    mag = hop_rows(waves)
    quad = hop_rows_quad(waves)
    assert quad.shape == (*mag.shape, 2)
    amplitude = quad.pow(2).sum(dim=-1).sqrt()
    assert torch.allclose(amplitude, mag, atol=1e-5)


def test_frontends_are_deterministic_and_batch_independent():
    torch.manual_seed(0)
    waves = torch.randn(3, 8000) * 0.2
    assert torch.equal(hop_rows(waves), hop_rows(waves))
    assert torch.allclose(hop_rows(waves[:1]), hop_rows(waves)[:1], atol=1e-6)
    assert torch.equal(hop_rows_quad(waves), hop_rows_quad(waves))

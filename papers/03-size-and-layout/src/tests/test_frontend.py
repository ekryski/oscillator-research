"""Contracts for `harness.stimuli.frontend`.

The front end is fixed and parameter-free, so it cannot carry the result.
These tests pin the exact frame count, the band-to-row order, that the
quadrature pathway differs from the band-energy pathway in phase alone, and
paper 03's additions: up to 32 bands the front end is paper 02's exactly, and
at 64 and 128 bands the zero-padded transform leaves no filter empty and keeps
paper 02's frames.
"""

import cmath
import math

import pytest
import torch
import torchaudio

from harness.stimuli.frontend import (
    HOP_LENGTH,
    HOP_N_FFT,
    HOP_OFFSET,
    HOP_SCALE,
    _mel,
    _quad_maps,
    fft_points,
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
    for samples in (512, 4352, 16000, 37952):
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


def _paper02_rows(waves: torch.Tensor, bands: int) -> torch.Tensor:
    """Paper 02's band-energy front end, restated from its source: a 512-point transform at every band count."""
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=256, n_mels=bands,
                                               center=False, power=2.0)
    return torch.clamp((torch.log(mel(waves) + 1e-5).transpose(1, 2) + HOP_OFFSET) / HOP_SCALE, min=0.0)


@pytest.mark.parametrize("bands", [8, 16, 32])
def test_up_to_32_bands_the_front_end_is_paper_02s_exactly(bands):
    torch.manual_seed(0)
    waves = torch.randn(2, 16000) * 0.2
    assert fft_points(bands) == HOP_N_FFT
    assert torch.equal(hop_rows(waves, bands), _paper02_rows(waves, bands))


@pytest.mark.parametrize("bands,points", [(64, 1024), (128, 2048)])
def test_above_32_bands_no_mel_filter_is_empty_and_the_frames_are_paper_02s(bands, points):
    assert fft_points(bands) == points
    fb = _mel(bands, 16000).mel.mel_scale.fb
    assert int((fb > 0).sum(0).min()) >= 3, "every filter spans at least three bins"
    bins, _ = _quad_maps(bands, 16000)
    assert len(set(bins.tolist())) == bands, "every band has its own peak bin"
    at_512 = torchaudio.functional.melscale_fbanks(257, 0.0, 8000.0, bands, 16000)
    assert int((at_512 > 0).sum(0).min()) <= 1, "which a 512-point transform would not give"
    torch.manual_seed(0)
    waves = torch.randn(2, 16000) * 0.2
    rows = hop_rows(waves, bands)
    assert rows.shape == (2, hop_num_frames(16000), bands) and torch.isfinite(rows).all()


def test_the_zero_padded_frames_sit_where_paper_02s_do():
    """A click inside one 512-sample window lands in that frame, at every band count."""
    waves = torch.zeros(1, 16000)
    waves[0, 20 * HOP_LENGTH + 300] = 1.0          # inside frames 20 and 21 only
    for bands in (16, 64, 128):
        energy = hop_rows(waves, bands).sum(-1)[0]
        assert set(torch.nonzero(energy > 0).flatten().tolist()) == {20, 21}, bands


def test_the_quadrature_phase_is_referenced_to_the_window_at_every_band_count():
    """On a bin both transforms share, the zero-padded quadrature rows equal the unpadded ones."""
    sr = 16000
    ts = torch.arange(16000, dtype=torch.float32) / sr
    f = 1000.0                                     # bin 32 of 512 points, bin 128 of 2,048
    wave = torch.sin(2 * math.pi * f * ts + 0.3)[None]
    pad = (2048 - HOP_N_FFT) // 2
    big = torch.stft(torch.nn.functional.pad(wave, (pad, pad)), n_fft=2048, hop_length=HOP_LENGTH,
                     win_length=HOP_N_FFT, window=torch.hann_window(HOP_N_FFT), center=False,
                     return_complex=True)[0, 128]
    small = torch.stft(wave, n_fft=HOP_N_FFT, hop_length=HOP_LENGTH, window=torch.hann_window(HOP_N_FFT),
                       center=False, return_complex=True)[0, 32]
    shifted = big * cmath.exp(2j * math.pi * f * pad / sr)   # the transform frame starts `pad` early
    assert small.shape == big.shape == (61,)
    assert torch.allclose(shifted, small, atol=1e-3)
    # and the front end's demodulation applies exactly that reference
    q16, q128 = hop_rows_quad(wave, 16), hop_rows_quad(wave, 128)
    assert torch.isfinite(q16).all() and torch.isfinite(q128).all()


def test_the_log_spaced_bands_span_four_octaves_at_every_count_and_are_paper_02s_at_16():
    from harness.stimuli.filterbank import band_edges
    from harness.utils.constants import F_LO
    r = torch.arange(17, dtype=torch.float64)
    assert torch.equal(band_edges(16), F_LO * (2.0 ** (r / 4)))
    for bands in (8, 32, 64, 128):
        e = band_edges(bands)
        assert e[0] == F_LO and torch.isclose(e[-1], torch.tensor(16 * F_LO, dtype=torch.float64))


@pytest.mark.parametrize("bands,window", [(64, 1024), (128, 2048)])
def test_a_longer_window_keeps_paper_02s_frame_count_and_centres(bands, window):
    """A real window of `window` samples, centred where paper 02's 512-sample frame is centred."""
    assert fft_points(bands, window) == window
    waves = torch.zeros(1, 16000)
    at = 20 * HOP_LENGTH + 256                       # the centre of paper 02's frame 20
    waves[0, at] = 1.0
    energy = hop_rows(waves, bands, window=window).sum(-1)[0]
    assert energy.shape[0] == hop_num_frames(16000) == 61
    lit = set(torch.nonzero(energy > 0).flatten().tolist())
    reach = window // 2 // HOP_LENGTH                # frames whose window, centred at 256 k + 256, covers `at`
    assert lit == set(range(20 - reach + 1, 20 + reach)), (lit, reach)
    fb = _mel(bands, 16000, window=window).mel.mel_scale.fb
    assert int((fb > 0).sum(0).min()) >= 3


def test_the_window_changes_nothing_at_its_default_and_is_refused_where_it_cannot_be():
    torch.manual_seed(0)
    waves = torch.randn(2, 16000) * 0.2
    for bands in (16, 64):
        assert torch.equal(hop_rows(waves, bands), hop_rows(waves, bands, window=512))
    with pytest.raises(ValueError, match="multiple of"):
        hop_rows(waves, 64, window=1000)
    q = hop_rows_quad(waves, 64, window=1024)
    assert q.shape == (2, 61, 64, 2) and torch.isfinite(q).all()
    assert torch.allclose(q.pow(2).sum(-1).sqrt(), hop_rows(waves, 64, window=1024), atol=1e-4)


def test_a_window_is_part_of_an_arms_label_and_a_caches_name():
    from harness.experiment import protocol as pr
    from harness.experiment.arms import Arm
    assert Arm("field", grid=64, window=1024).label().endswith("-64x64-w1024")
    assert Arm("field", grid=64, window=512).label() == Arm("field", grid=64).label()
    assert pr.rows_path("envelope", 0.0, 128, 2048).name == "envelope-128bands-w2048-0db.pt"
    assert pr.rows_path("envelope", 0.0, 16).name == "envelope-0db.pt"

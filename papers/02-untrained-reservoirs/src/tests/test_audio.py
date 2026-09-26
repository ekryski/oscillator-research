"""Contracts for `harness.audio` — the log-mel front-end.

Two properties matter for attribution and are asserted here. The frame count
must be exact, because every length mask downstream is computed from it rather
than measured. And nothing may depend on having seen the whole clip: an
utterance-level statistic is unknowable mid-stream, so a frontend that used one
would quietly make the pipeline non-streaming and would let the bookend do work
the field is being credited for.
"""

import sys

import torch

sys.path.insert(0, ".")
from harness.stimuli.audio import MelFrontend


def test_frame_count_is_exact_and_matches_the_declared_formula():
    fe = MelFrontend(sample_rate=16000, n_fft=512, hop=256, n_mels=16)
    for length in (512, 1000, 4096, 16000):
        expected = (length - 512) // 256 + 1
        assert fe(torch.randn(1, length)).shape[1] == expected
        assert int(fe.num_frames(torch.tensor([length]))[0]) == expected


def test_short_input_yields_no_frames_rather_than_a_negative_count():
    fe = MelFrontend(sample_rate=16000, n_fft=512, hop=256, n_mels=16)
    assert int(fe.num_frames(torch.tensor([100]))[0]) == 0


def test_output_shape_and_finiteness():
    fe = MelFrontend(sample_rate=16000, n_fft=512, hop=256, n_mels=16)
    out = fe(torch.randn(3, 8000))
    assert out.shape == (3, (8000 - 512) // 256 + 1, 16)
    assert torch.isfinite(out).all()


def test_no_trainable_parameters():
    """The frontend is FIXED by design so bookends cannot dominate attribution."""
    fe = MelFrontend(sample_rate=16000, n_fft=512, hop=256, n_mels=16)
    assert [p for p in fe.parameters() if p.requires_grad] == []


def test_is_deterministic_and_batch_independent():
    fe = MelFrontend(sample_rate=16000, n_fft=512, hop=256, n_mels=16)
    waves = torch.randn(4, 8000)
    a, b = fe(waves), fe(waves)
    assert torch.equal(a, b)
    # no per-utterance or per-batch statistic: one clip's features are the same
    # whether or not it travelled with company
    assert torch.allclose(fe(waves[:1]), a[:1], atol=1e-6)


def test_scaling_a_clip_shifts_log_mel_by_a_constant():
    """log(mel(a*x)) = log(mel(x)) + 2*log(a) up to the epsilon floor — the
    property that lets the fixed downstream affine stand in for normalization."""
    fe = MelFrontend(sample_rate=16000, n_fft=512, hop=256, n_mels=16)
    x = torch.randn(1, 8000) * 0.3
    delta = fe(4 * x) - fe(x)
    assert torch.allclose(delta, torch.full_like(delta, delta.median()), atol=0.05)


def test_lens_mask_zeroes_padded_frames_only():
    fe = MelFrontend(sample_rate=16000, n_fft=512, hop=256, n_mels=16)
    waves = torch.randn(2, 8000)
    lens = torch.tensor([8000, 2000])
    out = fe(waves, lens)
    keep = int(fe.num_frames(lens)[1])
    assert out[1, keep:].abs().sum() == 0
    assert out[1, :keep].abs().sum() > 0
    assert torch.equal(out[0], fe(waves)[0])  # the full-length row is untouched

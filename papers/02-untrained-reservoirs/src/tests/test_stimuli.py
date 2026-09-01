"""Contracts for `harness.stimuli`.

The stimuli and the fixed injection path are where a task's guarantees are
actually made: that a tone's energy lands in the row the filterbank claims,
that the two step orders have identical average spectra, that the drive is
broadcast identically across channels and columns, and that a digit pair's
two orders differ only in order."""

import math
import sys

import pytest
import torch

sys.path.insert(0, ".")
from harness.measurement.floor import floor_features
from harness.models import (
    OscillatorField,
)
from harness.stimuli import (
    PAIR_LEADER_SAMPLES,
    PAIR_MAX_SAMPLES,
    band_edges,
    band_index,
    bandpass_rows,
    drive_kick_stats,
    make_step_clips,
    quad_rows_to_drive,
    rows_to_drive,
    tone_classes,
)
from harness.utils.constants import WARMUP_FRAMES

TWO_PI = 2 * math.pi


def test_tone_energy_lands_in_its_band():
    # bin-aligned frequency inside row 9 -> no spectral leakage -> >=95% in-band
    t, grid = 512, 16
    e = band_edges(grid)
    k = round(float(math.sqrt(e[9] * e[10])) * t)  # nearest exact FFT bin
    f = k / t
    assert band_index(f, grid) == 9
    wave = torch.sin(TWO_PI * f * torch.arange(t))[None, :]
    rows = bandpass_rows(wave, grid)
    energy = rows.pow(2).sum(dim=1).squeeze(0)
    assert energy[9] / energy.sum() > 0.95


def test_tone_classes_pair_within_band():
    classes = tone_classes(16)
    assert len(classes) == 8
    for (f1, r1), (f2, r2) in zip(classes[::2], classes[1::2], strict=True):
        assert r1 == r2  # the within-band pair: same spatial row, different frequency
        assert band_index(f1, 16) == r1 and band_index(f2, 16) == r2
        assert f2 / f1 > 1.03  # separated by more than the +-1.5% frequency jitter


def test_step_stimuli_average_spectra_match():
    # A->B and B->A must be indistinguishable by average spectrum: the order
    # task is solvable only through temporal dynamics.
    gen = torch.Generator().manual_seed(0)
    wave, _, labels = make_step_clips(128, 512, 0.0156, 0.037, gen,
                                      noise_db=None, switch_jitter=0)
    spec = torch.fft.rfft(wave, dim=1).abs()
    m0, m1 = spec[labels == 0].mean(dim=0), spec[labels == 1].mean(dim=0)
    assert (m0 - m1).norm() / m0.norm() < 0.08


def test_drive_broadcast_shape_and_content():
    rows = torch.randn(2, 40, 16)
    drive = rows_to_drive(rows, channels=4, gain=2.0)
    assert drive.shape == (2, 40, 4, 16, 16)
    assert torch.equal(drive[:, :, 0], drive[:, :, 3])  # identical across channels
    assert torch.equal(drive[..., 0], drive[..., 15])  # identical across columns
    assert torch.allclose(drive[0, :, 0, 5, 0], 2.0 * rows[0, :, 5])


def test_am_clips_share_carrier_row_and_differ_only_in_envelope_rate():
    from harness.stimuli.synthetic import AM_CARRIER_ROW, am_rates, make_am_clips
    t, grid = 512, 16
    rates = am_rates()
    gen = torch.Generator().manual_seed(0)
    wave, phase, labels = make_am_clips(64, t, rates, grid, gen, noise_db=None)
    assert wave.shape == (64, t) and labels.max() == len(rates) - 1
    assert torch.allclose(wave.pow(2).mean(dim=1).sqrt(), torch.ones(64), atol=1e-4)
    # anti-spatial-cheat contract: the carrier row dominates energy for EVERY class
    rows = bandpass_rows(wave, grid)
    energy = rows.pow(2).sum(dim=1)
    assert (energy.argmax(dim=1) == AM_CARRIER_ROW).float().mean() > 0.95
    # envelope rate is recoverable: |wave| spectrum peaks at ~the class rate
    env = wave.abs()
    spec = torch.fft.rfft(env - env.mean(dim=1, keepdim=True), dim=1).abs()
    peak = spec.argmax(dim=1).float() / t
    want = torch.tensor(rates)[labels]
    ok = ((peak - want).abs() / want < 0.25).float().mean()
    assert ok > 0.8, f"envelope-rate recovery too weak: {ok:.2f}"


def test_quad_rows_drive_field_and_differ_from_magnitude():
    # quad frontend through the harness: [B,T,G,2] rows route to the Adler
    # drive; feature contract unchanged (they read theta); dynamics genuinely
    # differ from the magnitude (additive) path on the same envelopes.
    from harness import quad_rows_to_drive
    torch.manual_seed(0)
    amp = torch.rand(2, 48, 16) * 0.5
    phi = torch.rand(2, 48, 16) * TWO_PI
    quad = torch.stack((amp * torch.cos(phi), amp * torch.sin(phi)), dim=-1)
    d = quad_rows_to_drive(quad, channels=2, gain=2.0)
    assert d.shape == (2, 48, 2, 16, 16, 2)
    assert torch.equal(d[:, :, 0], d[:, :, 1]) and torch.equal(d[..., 0, :], d[..., 15, :])
    assert torch.allclose(d[0, :, 0, 5, 0, 0], 2.0 * quad[0, :, 5, 0])
    torch.manual_seed(6)
    m = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0)
    f_quad = m.features(quad)
    f_mag = m.features(amp)
    assert f_quad.shape == (2, m.feat_dim) and torch.isfinite(f_quad).all()
    assert not torch.allclose(f_quad, f_mag, atol=1e-3)
    theta = m.phase_trajectory(quad)  # instrumentation path accepts quad rows
    assert theta.shape == (2, 48, 2, 16, 16) and torch.isfinite(theta).all()
    # masked features work on quad rows too (digits contract)
    tv = torch.tensor([30, 48])
    assert torch.isfinite(m.features(quad, tvalid=tv)).all()


def test_drive_kick_stats_contract():
    # integrator-validity instrument: known rows -> exact stats
    from harness import drive_kick_stats

    rows = torch.full((2, 10, 4), 0.5)
    st = drive_kick_stats(rows, gain=2.0, dt=0.1)
    assert abs(st["kick_rms"] - 0.1) < 1e-6      # 0.1 * 2 * 0.5
    assert abs(st["kick_max"] - 0.1) < 1e-6
    assert st["kick_p999"] <= st["kick_max"]


def test_digitpair_clips_contract():
    #: order labels balanced-ish, floor-blindness by construction
    import pytest

    from harness.stimuli.digits import (
        DIGIT_BANK_PATH,
        PAIR_MAX_SAMPLES,
        load_digit_bank,
        make_digitpair_clips,
    )
    if not DIGIT_BANK_PATH.exists():
        pytest.skip("digit bank not built")
    bank = load_digit_bank()
    g = torch.Generator().manual_seed(5)
    w, lens, y = make_digitpair_clips(64, bank, "train", g, (3, 7), noise_db=None)
    assert w.shape == (64, PAIR_MAX_SAMPLES) and set(y.tolist()) == {0, 1}
    assert 10 <= int(y.sum()) <= 54  # both orders present
    assert (lens > 16000).all() and (lens <= PAIR_MAX_SAMPLES).all()
    # floor-blindness AFTER the warmup cut (the audit-matched window): the
    # leader absorbs the cut, so [A;B] vs [B;A] stay near-identical
    from harness import PAIR_LEADER_SAMPLES, WARMUP_FRAMES
    from harness.stimuli.frontend import hop_rows
    wa = bank["train"]["waves"][0].to(torch.float32) / 32767.0
    wb = bank["train"]["waves"][1].to(torch.float32) / 32767.0
    import torch as th
    L = PAIR_LEADER_SAMPLES
    ab = th.zeros(1, PAIR_MAX_SAMPLES)
    ba = th.zeros(1, PAIR_MAX_SAMPLES)
    ab[0, L:L + len(wa)] = wa
    ab[0, L + len(wa) + 1600:L + len(wa) + 1600 + len(wb)] = wb
    ba[0, L:L + len(wb)] = wb
    ba[0, L + len(wb) + 1600:L + len(wb) + 1600 + len(wa)] = wa
    ra = hop_rows(ab, 16)[:, WARMUP_FRAMES:]
    rb = hop_rows(ba, 16)[:, WARMUP_FRAMES:]
    fa, fb = floor_features(ra), floor_features(rb)
    assert (fa[0, :32] - fb[0, :32]).abs().max() < 0.02


def test_digit_bank_contract():
    # /: speaker-disjoint, true lengths, deterministic sampling
    import pytest

    from harness.stimuli.digits import DIGIT_BANK_PATH, load_digit_bank, make_digit_clips
    if not DIGIT_BANK_PATH.exists():
        pytest.skip("digit bank not built")
    bank = load_digit_bank()
    tr = set(bank["train"]["speakers"].tolist())
    te = set(bank["test"]["speakers"].tolist())
    assert not (tr & te) and len(tr) == 48 and len(te) == 12
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    w1, l1, y1 = make_digit_clips(32, bank, "test", g1, noise_db=-10.0)
    w2, l2, y2 = make_digit_clips(32, bank, "test", g2, noise_db=-10.0)
    assert torch.equal(w1, w2) and torch.equal(y1, y2)  # deterministic
    assert w1.shape == (32, 16000) and (l1 <= 16000).all() and (l1 > 1600).all()
    assert torch.isfinite(w1).all() and y1.min() >= 0 and y1.max() <= 9


def test_band_edges_are_log_spaced_at_four_bands_per_octave():
    e = band_edges(16)
    assert e.shape == (17,)
    assert torch.all(e[1:] > e[:-1])
    ratios = (e[1:] / e[:-1]).double()
    assert torch.allclose(ratios, torch.full_like(ratios, 2 ** 0.25), atol=1e-9)
    assert e[-1] / e[0] == pytest.approx(16.0), "16 rows span exactly four octaves"


def test_band_index_inverts_band_edges():
    e = band_edges(16)
    for r in range(16):
        centre = float((e[r] * e[r + 1]).sqrt())
        assert band_index(centre, 16) == r


def test_bandpass_rows_partition_the_in_band_energy():
    """The band split is a disjoint FFT mask, so summing the rows reconstructs
    the in-band signal and no energy is double-counted."""
    torch.manual_seed(0)
    wave = torch.randn(2, 512)
    rows = bandpass_rows(wave, 16)
    assert rows.shape == (2, 512, 16)
    summed = rows.sum(dim=2)
    # the sum is the band-limited signal: it can only lose energy, never gain
    assert (summed.pow(2).sum() <= wave.pow(2).sum() + 1e-3)
    per_row = rows.pow(2).sum(dim=(0, 1))
    assert torch.allclose(per_row.sum(), summed.pow(2).sum(), rtol=1e-3)


def test_drive_kick_stats_scale_linearly_with_gain_and_dt():
    rows = torch.rand(4, 64, 16)
    a = drive_kick_stats(rows, gain=1.0, dt=0.1)
    b = drive_kick_stats(rows, gain=2.0, dt=0.1)
    c = drive_kick_stats(rows, gain=1.0, dt=0.2)
    for key in ("kick_rms", "kick_max", "kick_p999"):
        assert b[key] == pytest.approx(2 * a[key], rel=1e-6)
        assert c[key] == pytest.approx(2 * a[key], rel=1e-6)
    assert a["kick_rms"] <= a["kick_p999"] <= a["kick_max"]


def test_integrator_validity_bound_is_the_max_criterion():
    """A run is valid only while every per-tick drive-phase increment stays
    below pi; the instrument has to report the MAX, not an average."""
    rows = torch.zeros(1, 8, 16)
    rows[0, 0, 0] = 100.0                      # one enormous sample
    stats = drive_kick_stats(rows, gain=1.0, dt=0.1)
    assert stats["kick_max"] == pytest.approx(10.0)
    assert stats["kick_max"] > math.pi, "the outlier must not be averaged away"
    assert stats["kick_rms"] < math.pi


def test_quad_and_magnitude_drives_share_their_broadcast_contract():
    rows = torch.rand(2, 10, 16)
    quad = torch.stack((rows, torch.zeros_like(rows)), dim=-1)
    mag_drive = rows_to_drive(rows, channels=4, gain=2.0)
    quad_drive = quad_rows_to_drive(quad, channels=4, gain=2.0)
    assert mag_drive.shape == (2, 10, 4, 16, 16)
    assert quad_drive.shape == (2, 10, 4, 16, 16, 2)
    assert torch.equal(quad_drive[..., 0], mag_drive)


def test_digit_pair_leader_absorbs_the_featurization_warmup():
    """The silent leader exists so the warmup cut consumes leader rather than
    the first digit's onset — without it the readout window is not
    order-symmetric and the 'order-free' floor leaks."""
    from harness.stimuli.frontend import HOP_LENGTH, hop_num_frames
    assert hop_num_frames(PAIR_LEADER_SAMPLES) >= WARMUP_FRAMES
    assert PAIR_LEADER_SAMPLES > WARMUP_FRAMES * HOP_LENGTH
    assert PAIR_MAX_SAMPLES > PAIR_LEADER_SAMPLES + 2 * 16000

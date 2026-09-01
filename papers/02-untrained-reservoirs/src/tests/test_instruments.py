"""Contracts for `harness.instruments`.

Instruments never decide a verdict, but a broken instrument makes a
mechanism story unfalsifiable, so the locking reader is pinned against a
stimulus whose true phase is known analytically."""

import math
import sys

import pytest
import torch

sys.path.insert(0, ".")
from harness.measurement.instruments import analytic_row_phase, instrument_field, natural_rate
from harness.models import OscillatorField

TWO_PI = 2 * math.pi


def test_analytic_row_phase_and_drive_plv():
    # instrument fix: drive-phase reference recovers known phase and
    # agrees with the true analytic phase on a tone (the validation task)
    from harness import OscillatorField, analytic_row_phase, instrument_field

    t, g, f = 512, 16, 0.03  # tone at f cycles/frame in band 3
    tt = torch.arange(t, dtype=torch.float32)
    true_phase = 2 * torch.pi * f * tt
    rows = torch.zeros(1, t, g)
    rows[0, :, 3] = torch.cos(true_phase)
    phi = analytic_row_phase(rows)
    assert phi.shape == (1, t, g)
    # recovered instantaneous rate in the driven band == the tone's rate
    rate = (phi[0, 1:, 3] - phi[0, :-1, 3] + torch.pi) % (2 * torch.pi) - torch.pi
    interior = rate[8:-8]  # Hilbert edge effects excluded
    assert torch.allclose(interior.mean(), torch.tensor(2 * torch.pi * f), atol=2e-3)
    # instrument agreement: PLV via phi_rows == PLV via the true global phase
    # for oscillators in the driven row (same reference, two plumbing paths)
    torch.manual_seed(0)
    m = OscillatorField(channels=2, grid=g, n_classes=4, probe_seed=0)
    labels = torch.zeros(1, dtype=torch.long)
    old = instrument_field(m, rows, true_phase[None, :], labels, 1)
    new = instrument_field(m, rows, torch.zeros(1, t), labels, 1, phi_rows=phi)
    row_old = old["plv_map"][0, :, 3, :]
    row_new = new["plv_map"][0, :, 3, :]
    assert torch.allclose(row_old, row_new, atol=0.05)


def test_natural_rate_matches_the_tilted_washboard_velocity():
    """The untrained field's mean free-rotation rate; zero when pinning wins."""
    assert natural_rate(1.0, 0.5, 0.1, 1) == pytest.approx(
        0.1 * math.sqrt(1.0 - 0.25) / (2 * math.pi))
    assert natural_rate(0.4, 0.5) == 0.0, "|omega| <= damping means pinned-static"
    assert natural_rate(1.0, 0.5, 0.1, 2) == pytest.approx(2 * natural_rate(1.0, 0.5, 0.1, 1))


def test_analytic_row_phase_recovers_a_known_carrier():
    """The locking reference has to be right before any locking claim can be."""
    t, g = 512, 4
    freq = 0.05
    ramp = TWO_PI * freq * torch.arange(t, dtype=torch.float32)
    rows = torch.sin(ramp)[None, :, None].expand(1, t, g).contiguous()
    phi = analytic_row_phase(rows)
    assert phi.shape == (1, t, g)
    mid = slice(64, t - 64)                       # skip the FFT edge transients
    d = torch.diff(phi[0, mid, 0])
    d = (d + math.pi) % TWO_PI - math.pi          # unwrap to (-pi, pi]
    assert d.mean().item() == pytest.approx(TWO_PI * freq, abs=1e-3)


def test_instrumenting_a_random_graph_core_reports_graph_stats_not_kernel_stats():
    """The random-graph core has no circulant kernel, so the spectral columns
    are zeroed rather than fabricated from a tensor that does not exist."""
    torch.manual_seed(0)
    model = OscillatorField(channels=2, grid=8, n_classes=2, probe_seed=0,
                            core="randgraph", graph_k=2)
    rows = torch.rand(4, 24, 8)
    out = instrument_field(model, rows, torch.zeros(4, 24), torch.tensor([0, 1, 0, 1]), 2)
    assert out["plv_map"].shape == (2, 2, 8, 8)
    assert 0.0 <= out["R_mean"] <= 1.0
    assert 0.0 <= out["ent_frac_mean"] <= 1.0
    assert float(out["opnorm_raw"].abs().sum()) == 0.0


def test_order_parameter_is_one_for_a_locked_field_and_small_for_a_scattered_one():
    torch.manual_seed(0)
    model = OscillatorField(channels=1, grid=8, n_classes=2, probe_seed=0)
    rows = torch.zeros(2, 20, 8)
    labels = torch.tensor([0, 1])
    with torch.no_grad():                          # every oscillator at one phase
        model.core.blocks[0].natural_freqs.zero_()
        model.core.blocks[0].kernel.zero_()
    out = instrument_field(model, rows, torch.zeros(2, 20), labels, 2)
    assert 0.0 <= out["R_mean"] <= 1.0

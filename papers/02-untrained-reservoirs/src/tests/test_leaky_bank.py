"""The non-oscillating control: matched to the field, and missing exactly what is under test."""

import pytest
import torch

from harness.models.field import OscillatorField
from harness.models.leaky_bank import HOP_RATE_HZ, TAU_MAX_S, TAU_MIN_S, LeakyBank


def rows(batch=2, frames=30, grid=16, seed=0):
    return torch.rand(batch, frames, grid, generator=torch.Generator().manual_seed(seed))


def count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def test_bank_a_matches_the_field_in_states_and_in_stored_parameters():
    field = OscillatorField(channels=4, grid=16, n_classes=10)
    bank = LeakyBank(channels=4, grid=16)
    assert bank.n_states == 4 * 16 * 16 == 1024
    assert count(bank) == count(field) == 2048


def test_bank_b_matches_the_field_in_exposed_signals_at_twice_the_parameters():
    # the field exposes sin and cos per oscillator, a leaky unit exposes one state
    field = OscillatorField(channels=4, grid=16, n_classes=10)
    bank = LeakyBank(channels=8, grid=16)
    with torch.no_grad():
        assert bank.signals(rows()).shape[2] == field._scan(rows()).shape[2] == 2048
    assert count(bank) == 4096


def test_nothing_in_it_is_trained():
    assert not any(p.requires_grad for p in LeakyBank().parameters())


def test_the_same_seed_gives_the_same_bank_and_a_different_seed_another():
    x = rows()
    assert torch.equal(LeakyBank(seed=3).signals(x), LeakyBank(seed=3).signals(x))
    assert not torch.equal(LeakyBank(seed=3).signals(x), LeakyBank(seed=4).signals(x))


def test_only_the_gains_vary_with_the_seed_so_its_seed_spread_is_small_by_construction():
    a, b = LeakyBank(seed=0), LeakyBank(seed=1)
    assert torch.equal(a.tau_s, b.tau_s) and not torch.equal(a.input_gain, b.input_gain)


def test_there_is_no_coupling_a_unit_answers_to_its_own_band_alone():
    bank, x = LeakyBank(), rows()
    other = x.clone()
    other[:, :, 5] = 9.0                                  # disturb band 5 only
    before, after = bank.signals(x), bank.signals(other)
    lattice = lambda s: s.view(*s.shape[:2], bank.channels, bank.grid, bank.grid)  # noqa: E731
    changed = (lattice(before) != lattice(after)).any(dim=0).any(dim=0)            # [C, row, col]
    assert changed[:, 5, :].all() and not changed[:, :5, :].any() and not changed[:, 6:, :].any()


def test_there_is_no_oscillation_a_step_is_approached_without_overshoot():
    bank = LeakyBank()
    step = torch.ones(1, 200, 16)
    x = bank.signals(step)[0]                              # [T, states]
    assert (x[1:] >= x[:-1] - 1e-7).all()                  # monotone rise
    ceiling = torch.tanh(bank.input_gain.reshape(-1) * bank.gain)
    assert (x <= ceiling + 1e-6).all()                     # never past its target


def test_every_band_is_read_at_every_time_scale_from_one_frame_to_a_whole_clip():
    bank = LeakyBank()
    assert bank.tau_s.min().item() == pytest.approx(TAU_MIN_S, rel=1e-5)
    assert bank.tau_s.max().item() == pytest.approx(TAU_MAX_S, rel=1e-5)
    per_row = bank.tau_s[:, 0, :].reshape(-1)               # one row's units
    assert per_row.numel() == 64 and per_row.unique().numel() == 64
    assert torch.equal(bank.tau_s[:, 0, :], bank.tau_s[:, 9, :])   # the same schedule in every row
    ratios = per_row.sort().values[1:] / per_row.sort().values[:-1]
    assert torch.allclose(ratios, ratios[0].expand_as(ratios), rtol=1e-4)   # log-spaced


def test_with_no_drive_the_state_carries_nothing_about_the_clip():
    # gain 0 is the sanity cell: the read must then sit at chance
    bank = LeakyBank(gain=0.0)
    assert torch.equal(bank.signals(rows(seed=1)), bank.signals(rows(seed=2)))
    assert bank.signals(rows()).abs().max().item() == 0.0


def test_time_constants_are_physical_so_the_bank_means_the_same_filters_at_any_rate():
    # a constant input, read after the same number of SECONDS at two frame rates
    slow, fast = LeakyBank(rate_hz=HOP_RATE_HZ), LeakyBank(rate_hz=4 * HOP_RATE_HZ)
    seconds = 0.4
    at_slow = slow.signals(torch.ones(1, int(seconds * HOP_RATE_HZ), 16))[0, -1]
    at_fast = fast.signals(torch.ones(1, int(seconds * 4 * HOP_RATE_HZ), 16))[0, -1]
    assert torch.allclose(at_slow, at_fast, atol=1e-5)


def test_the_quadrature_pathway_is_refused_because_a_leaky_unit_has_no_phase():
    with pytest.raises(ValueError, match="no phase"):
        LeakyBank().signals(torch.rand(2, 10, 16, 2))


def test_a_front_end_with_the_wrong_number_of_bands_is_refused():
    with pytest.raises(ValueError, match="bands"):
        LeakyBank(grid=16).signals(torch.rand(2, 10, 8))

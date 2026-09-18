"""The shared readout contract: what every arm's features may and may not depend on."""

import pytest
import torch

from harness.measurement import features as ft


def trajectory(batch=3, frames=20, dims=4, seed=0):
    return torch.randn(batch, frames, dims, generator=torch.Generator().manual_seed(seed))


def test_the_three_statistics_are_what_they_say():
    x = torch.tensor([[[1.0], [3.0], [2.0], [6.0]]])
    mean, std, change = ft.pooled(x)[0].tolist()
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(torch.tensor([1.0, 3.0, 2.0, 6.0]).std().item())
    assert change == pytest.approx((2 + 1 + 4) / 3)


def test_width_is_three_per_signal_per_window():
    x = trajectory(dims=5)
    assert ft.pooled(x).shape == (3, ft.STATS_PER_SIGNAL * 5)
    assert ft.windowed(x, windows=4).shape == (3, 4 * ft.STATS_PER_SIGNAL * 5)


def test_frames_outside_the_span_cannot_touch_the_read():
    # padding after a clip ends, and a warm-up before it starts, must be invisible:
    # the exploratory floor pooled over both and every other arm over neither
    x = trajectory()
    hi = torch.tensor([12, 15, 9])
    clean = ft.windowed(x, 2, lo=4, hi=hi)
    dirty = x.clone()
    dirty[:, :4] = 99.0
    for i, h in enumerate(hi):
        dirty[i, h:] = -77.0
    assert torch.equal(ft.windowed(dirty, 2, lo=4, hi=hi), clean)


def test_no_feature_is_an_endpoint():
    # a recurrent net's last state names the digit it heard last, which is order itself
    x = torch.zeros(1, 10, 2)
    x[0, -1] = 1000.0
    assert not (ft.pooled(x) == 1000.0).any()
    x[0, 0] = -1000.0
    assert not (ft.pooled(x).abs() == 1000.0).any()


def test_a_rising_state_and_a_falling_one_read_the_same():
    # the signed mean of a first difference telescopes to (last - first) / T: an
    # endpoint read in disguise. The absolute change used here cannot tell them apart.
    ramp = torch.linspace(0, 1, 12)[None, :, None]
    assert torch.allclose(ft.pooled(ramp), ft.pooled(ramp.flip(1)))


def test_the_whole_span_read_is_blind_to_order_except_at_the_junction():
    # "a then b" against "b then a": mean and spread are functions of the multiset
    # of frames, so they match exactly; only the mean change sees the one junction
    a, b = trajectory(1, 8, 3, seed=1), trajectory(1, 8, 3, seed=2)
    ab, ba = ft.pooled(torch.cat((a, b), 1)), ft.pooled(torch.cat((b, a), 1))
    d = 3
    assert torch.allclose(ab[:, :2 * d], ba[:, :2 * d], atol=1e-6)
    assert not torch.allclose(ab[:, 2 * d:], ba[:, 2 * d:])


def test_a_windowed_read_is_not_order_blind_which_is_why_order_is_scored_whole_span():
    a, b = trajectory(1, 8, 3, seed=1), trajectory(1, 8, 3, seed=2)
    ab, ba = ft.windowed(torch.cat((a, b), 1), 2), ft.windowed(torch.cat((b, a), 1), 2)
    assert not torch.allclose(ab[:, :3], ba[:, :3])


def test_each_window_is_the_pooled_read_of_its_own_slice():
    x = trajectory(1, 16, 2)
    w = ft.windowed(x, 4)
    for j in range(4):
        assert torch.allclose(w[:, j * 6:(j + 1) * 6], ft.pooled(x[:, 4 * j:4 * (j + 1)]), atol=1e-6)


def test_the_vectorized_windows_match_a_clip_by_clip_reference():
    x, lo, windows = trajectory(5, 30, 3, seed=3), 6, 4
    hi = torch.tensor([30, 22, 17, 14, 29])
    fast = ft.windowed(x, windows, lo=lo, hi=hi)
    for i in range(x.shape[0]):
        h = int(hi[i])
        parts = [ft.pooled(x[i:i + 1, lo + (h - lo) * j // windows: lo + (h - lo) * (j + 1) // windows])
                 for j in range(windows)]
        assert torch.allclose(fast[i:i + 1], torch.cat(parts, 1), atol=1e-5)


def test_a_clip_too_short_for_its_windows_is_widened_to_the_minimum_for_every_arm_alike():
    x = trajectory(2, 20, 2)
    short, enough = torch.tensor([5, 5]), torch.tensor([4 + 2 * 3, 4 + 2 * 3])
    assert torch.equal(ft.windowed(x, 3, lo=4, hi=short), ft.windowed(x, 3, lo=4, hi=enough))


def test_a_span_that_cannot_hold_the_windows_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="cannot hold"):
        ft.windowed(trajectory(1, 6, 1), windows=4, lo=2)


def test_rotation_rate_reads_a_steady_rotation_as_its_rate():
    omega, t = 0.3, torch.arange(25.0)
    theta = omega * t[None, :, None] + torch.tensor([0.0, 1.0])[None, None, :]
    rate = ft.rotation_rate(torch.cat((theta.sin(), theta.cos()), dim=2))
    assert torch.allclose(rate[0, :2], torch.full((2,), torch.cos(torch.tensor(omega)).item()), atol=1e-5)
    assert torch.allclose(rate[0, 2:], torch.full((2,), torch.sin(torch.tensor(omega)).item()), atol=1e-5)


def test_rotation_rate_counts_only_pairs_wholly_inside_the_span():
    theta = torch.zeros(1, 10, 1)
    theta[0, :5] = 0.0
    theta[0, 5:] = 2.0            # the only rotation is the jump between frames 4 and 5
    sincos = torch.cat((theta.sin(), theta.cos()), dim=2)
    inside = ft.rotation_rate(sincos, lo=5)      # span starts after the jump
    assert torch.allclose(inside, torch.tensor([[1.0, 0.0]]), atol=1e-6)
    assert not torch.allclose(ft.rotation_rate(sincos, lo=4), torch.tensor([[1.0, 0.0]]), atol=1e-3)


def test_rotation_rate_width_is_two_per_oscillator_per_window():
    sincos = trajectory(2, 24, 10)
    assert ft.rotation_rate(sincos, windows=4).shape == (2, 4 * 10)


def test_an_arm_is_never_read_wider_than_it_is():
    f = torch.randn(4, 192)
    assert ft.project(f, 192) is f and ft.project(f, 1024) is f


def test_the_projection_is_fixed_so_every_arm_and_machine_gets_the_same_one():
    f = torch.randn(4, 500, generator=torch.Generator().manual_seed(0))
    first = ft.project(f, 72)
    ft._PROJECTIONS.clear()                       # a fresh process draws the same matrix
    assert first.shape == (4, 72) and torch.equal(ft.project(f, 72), first)
    assert not torch.allclose(ft.project(f, 96)[:, :72], first)

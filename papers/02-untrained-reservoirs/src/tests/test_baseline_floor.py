"""Contracts for `harness.measurement.floor` — the no-dynamics floor."""

import math
import sys

import pytest
import torch

sys.path.insert(0, ".")
from harness.measurement.floor import floor_features, floor_features_windowed

TWO_PI = 2 * math.pi


def test_floor_features_windowed_parity_shape():
    # floor windowed-parity features: windows x 3G, defined for short windows >= 2
    rows = torch.randn(4, 64, 16)
    assert floor_features(rows).shape == (4, 48)
    assert floor_features_windowed(rows, 4).shape == (4, 192)


def test_floor_features_are_the_declared_three_band_statistics():
    rows = torch.zeros(2, 20, 16)
    rows[0] = 1.0                                  # constant: zero std, zero delta
    f = floor_features(rows)
    assert f.shape == (2, 48)
    mean, std, delta = f.split(16, dim=1)
    assert torch.allclose(mean[0], torch.ones(16))
    assert torch.allclose(std[0], torch.zeros(16), atol=1e-6)
    assert torch.allclose(delta[0], torch.zeros(16), atol=1e-6)


def test_the_floor_has_no_dynamics_and_no_parameters():
    """Whatever an arm scores above this line is what its dynamics bought, so
    the floor itself must be a pure function of the frontend rows."""
    torch.manual_seed(0)
    rows = torch.rand(4, 30, 16)
    assert torch.equal(floor_features(rows), floor_features(rows))
    assert torch.allclose(floor_features(rows[:1]), floor_features(rows)[:1])


def test_windowed_floor_is_the_per_window_concatenation():
    torch.manual_seed(0)
    rows = torch.rand(3, 40, 16)
    w = floor_features_windowed(rows, 4)
    assert w.shape == (3, 4 * 48)
    assert torch.allclose(w[:, :48], floor_features(rows[:, :10]))


def test_windowed_floor_refuses_windows_too_short_to_define_a_std():
    with pytest.raises(AssertionError, match="window too short"):
        floor_features_windowed(torch.rand(1, 6, 16), 4)

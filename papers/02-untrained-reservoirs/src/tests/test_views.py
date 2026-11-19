"""The views of a network's signals: phase alone, amplitude alone, and a slaved amplitude."""

import torch

from harness.experiment import arms as am
from harness.experiment import views as vw
from harness.experiment.arms import Arm


def test_the_phase_view_has_unit_radius_and_the_amplitude_view_is_the_radius():
    gen = torch.Generator().manual_seed(0)
    n = 2 * 4 * 4                                   # a 2-channel 4 x 4 network
    theta = torch.rand(3, 5, n, generator=gen) * 6.28
    r = 0.2 + torch.rand(3, 5, n, generator=gen)
    signals = torch.cat((r * torch.sin(theta), r * torch.cos(theta)), -1)
    sl = Arm("network", coupling="stuart-landau")
    phase = vw.apply_view(sl, None, signals, "phase")
    assert torch.allclose(phase[..., :n] ** 2 + phase[..., n:] ** 2, torch.ones(3, 5, n), atol=1e-5)
    assert torch.allclose(vw.apply_view(sl, None, signals, "amplitude"), r, atol=1e-5)
    assert torch.equal(vw.apply_view(sl, None, signals, "full"), signals)
    assert torch.equal(vw.apply_view(Arm("baseline"), None, signals, "phase"), signals)   # only networks have a z


def test_the_slaved_amplitude_follows_the_kuramoto_phases_and_stays_near_one():
    arm = Arm("network")
    model = am.build_untrained(arm, 1.0, 0)
    gen = torch.Generator().manual_seed(1)
    rows = torch.rand(2, 12, am.GRID, generator=gen)
    with torch.no_grad():
        signals = am.untrained_signals(arm, model, rows)
        out = vw.apply_view(arm, model, signals, "slaved")
    n = signals.shape[-1] // 2
    assert out.shape == signals.shape
    r = torch.sqrt(out[..., :n] ** 2 + out[..., n:] ** 2)
    assert (r > 0).all() and (r < 3).all() and 0.5 < r.mean() < 1.5   # the core's own radii spread this widely
    # the phases are the network's own: the scaled signals point the same way
    assert torch.allclose(out[..., :n] / r, signals[..., :n], atol=1e-4)
    assert torch.allclose(out[..., n:] / r, signals[..., n:], atol=1e-4)


def test_the_stiffness_names_the_arm_and_sets_the_relaxation_rate():
    arm = Arm("network", coupling="stuart-landau-exact", stiffness=16.0)
    assert arm.label().endswith("-stiffness16")
    assert "stiffness" not in Arm("network", coupling="stuart-landau-exact").label()
    model = am.build_untrained(arm, 1.0, 0)
    assert torch.allclose(model.core.alpha, torch.full_like(model.core.alpha, 0.8))
    beta = torch.nn.functional.softplus(model.core.beta_hat)
    assert torch.allclose(beta, model.core.alpha, atol=1e-5)          # the attracting radius stays 1

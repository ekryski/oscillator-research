"""The reservoir baselines: the echo state network, the random oscillators network and the memoryless expansion."""

import torch

from harness.experiment import arms as am
from harness.experiment import terms
from harness.models.reservoirs import EchoStateNetwork, RandomOscillators, _spectral_radius


def rows(batch=2, frames=30, grid=16, seed=0):
    return torch.rand(batch, frames, grid, generator=torch.Generator().manual_seed(seed))


def test_the_recurrent_matrix_has_the_asked_spectral_radius_and_the_asked_fan_in():
    esn = EchoStateNetwork(units=256, spectral_radius=0.9, seed=1)
    assert abs(_spectral_radius(esn.w) - 0.9) < 1e-4
    assert abs((esn.w != 0).float().sum(1).mean().item() - 10) < 2       # about ten incoming connections
    ron = RandomOscillators(units=128, spectral_radius=3.0, seed=1)
    assert abs(_spectral_radius(ron.w) - 3.0) < 1e-4


def test_nothing_in_them_is_trained_and_the_seed_pins_them():
    x = rows()
    for make in (lambda s: EchoStateNetwork(units=64, seed=s), lambda s: RandomOscillators(units=32, seed=s)):
        assert not any(p.requires_grad for p in make(0).parameters())
        assert torch.equal(make(3).signals(x), make(3).signals(x))
        assert not torch.equal(make(3).signals(x), make(4).signals(x))


def test_sizes_match_the_oscillator_network():
    assert EchoStateNetwork(units=1024).n_states == 1024
    assert RandomOscillators(units=512).n_states == 1024
    with torch.no_grad():
        assert EchoStateNetwork(units=1024).signals(rows()).shape == (2, 30, 1024)
        assert RandomOscillators(units=512).signals(rows()).shape == (2, 30, 1024)


def test_the_expansion_has_no_memory_and_the_echo_state_network_does():
    x = rows()
    other = x.clone()
    other[:, 10] += 1.0                                              # disturb one frame
    memoryless = EchoStateNetwork(units=64, spectral_radius=0.0, leak=1.0, seed=0)
    esn = EchoStateNetwork(units=64, spectral_radius=0.9, leak=0.3, seed=0)
    with torch.no_grad():
        d0 = (memoryless.signals(x) - memoryless.signals(other)).abs().sum(2)[0]   # [T]
        d1 = (esn.signals(x) - esn.signals(other)).abs().sum(2)[0]
    assert d0[10] > 0 and d0[:10].sum() == 0 and d0[11:].sum() == 0  # the expansion answers to its own frame alone
    assert d1[11:].sum() > 0 and d1[:10].sum() == 0                  # the network carries the disturbance forward


def test_the_random_oscillators_oscillate_and_settle_without_input():
    ron = RandomOscillators(units=16, tau=0.1, spectral_radius=0.0, gamma=(1.0, 0.0), eps=(0.2, 0.0), seed=0)
    with torch.no_grad():
        ron.b.zero_()
        y = torch.ones(1, 16)
        z = torch.zeros(1, 16)
        trace = []
        for _ in range(400):
            z = z + 0.1 * (-ron.gamma * y - ron.eps * z)
            y = y + 0.1 * z
            trace.append(y[0, 0].item())
    trace = torch.tensor(trace)
    assert trace.min() < -0.3                                  # it swings through zero: an oscillation
    assert trace[-50:].abs().max() < 0.1                       # and damps out: fading memory


def test_the_arms_build_label_and_name_themselves():
    esn = am.Arm("esn", units=1024, rho=0.9, leak=0.3)
    ron = am.Arm("ron", units=512, tau=0.1, rho=3.0, gamma=(1.0, 0.5), eps=(0.5, 0.4))
    exp = am.Arm("expansion", units=2048)
    assert esn.label() == "esn-n1024-rho0.9-leak0.3"
    assert ron.label() == "ron-n512-tau0.1-rho3-gamma1r0.5-eps0.5r0.4"
    assert exp.label() == "expansion-n2048"
    assert terms.arm(esn.label()) == "echo state network (1,024 units, ρ 0.9, leak 0.3)"
    assert terms.arm(ron.label()) == ("random oscillators network (512 oscillators, τ 0.1, ρ 3, "
                                      "γ 1 ± 0.5, ε 0.5 ± 0.4)")
    assert terms.arm(exp.label()) == "random nonlinear expansion (2,048 units)"
    for arm in (esn, ron, exp):
        assert arm.uses_gain
        model = am.build_untrained(arm, 1.0, 0)
        with torch.no_grad():
            sig = am.untrained_signals(arm, model, rows())
        assert sig.shape[:2] == (2, 30) and torch.isfinite(sig).all()
        assert am.meta(arm, model)["states"] == (1024 if arm.kind != "expansion" else 2048)
        assert am.reads(arm, "recognition") == {"windowed": ["windowed"], "pooled": ["pooled"]}

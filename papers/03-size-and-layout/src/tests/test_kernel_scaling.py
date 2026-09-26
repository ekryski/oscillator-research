"""Every channel's coupling kernel is scaled to exactly the coupling ceiling, and at 16 x 16 that is paper 02.

Paper 02 scaled a channel's kernel down to the ceiling when the kernel's
spectral peak exceeded it, and left it alone otherwise (`kernel_scaling="cap"`).
Paper 03 scales every kernel so its peak is the ceiling (`"exact"`). Where the
ceiling binds, the two are one computation: the cap is a clamp at 1 that does
not bind, and multiplying by the same factor gives the same bits. A random
16 x 16 kernel's peak is 1.42 or more on every geometry, seed and channel
paper 02 or paper 03 draws, above both ceilings paper 02 used (1 and 0.5), so
every 16 x 16 network is bit-identical under the two rules, and paper 02's
16 x 16 runs stand as paper 03's. On an 8 x 8 lattice the peak is usually below
1 (median 0.80), which is where the rules differ, and why paper 03 changes it.
"""

import pytest
import torch

from harness.experiment import arms as am
from harness.experiment import plan
from harness.models.field import physics_block
from harness.models.geometries import BOUNDARIES
from harness.models.phase import PhaseBlock, ceiling_factor

SEEDS = (0, 1, 2)


def _arms_16():
    """Every 16 x 16 network either paper runs: every coupling function, geometry (the coil and the
    cochleas among them), channel count and ceiling, and paper 02's restoring strengths and natural
    frequencies (they draw nothing extra)."""
    cochlea = [(c, g) for g in plan.COCHLEA_GEOMETRIES for c in plan.PHASE_COUPLINGS]
    for coupling, geometry in [*plan.designs(), *cochlea]:
        for channels in plan.CHANNELS:
            for ceiling in (1.0, 0.5):
                yield am.Arm("network", coupling=coupling, geometry=geometry, channels=channels, ceiling=ceiling)


def _operator(arm: am.Arm, seed: int, scaling: str, impl: str) -> torch.Tensor:
    net = am.build_untrained(arm, 1.0, seed, kernel_scaling=scaling)
    block = physics_block(net.core)
    if hasattr(block, "coupling_impl"):
        block.coupling_impl = impl
    coup = block.prepare_coupling()
    return torch.view_as_real(coup[0]) if isinstance(coup, tuple) else (
        torch.view_as_real(coup) if coup.is_complex() else coup)


@pytest.mark.parametrize("arm", list(_arms_16()), ids=lambda a: a.label())
def test_at_16x16_exact_scaling_is_paper_02s_bit_for_bit(arm):
    for seed in SEEDS:
        net = am.build_untrained(arm, 1.0, seed)
        assert (physics_block(net.core).peaks() > arm.ceiling).all(), "the ceiling binds on every channel"
        for impl in ("matmul", "fft"):
            exact, cap = _operator(arm, seed, "exact", impl), _operator(arm, seed, "cap", impl)
            assert torch.equal(exact, cap), (arm.label(), seed, impl)


@pytest.mark.parametrize("coupling,geometry", [("kuramoto", "torus"), ("winfree", "sheet"), ("stuart-landau", "torus")])
def test_at_16x16_the_whole_trajectory_is_paper_02s_bit_for_bit(coupling, geometry):
    arm = am.Arm("network", coupling=coupling, geometry=geometry)
    rows = torch.rand(3, 30, 16, generator=torch.Generator().manual_seed(4)) * 1.5
    with torch.no_grad():
        exact = am.build_untrained(arm, 2.0, 1)._scan(rows)
        cap = am.build_untrained(arm, 2.0, 1, kernel_scaling="cap")._scan(rows)
    assert torch.equal(exact, cap)


@pytest.mark.parametrize("grid", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_every_channel_meets_the_ceiling_exactly(boundary, grid):
    torch.manual_seed(0)
    blk = PhaseBlock(4, grid, spectral_clamp=1.0, boundary=boundary, coupling_impl="fft")
    scaled = blk.prepare_coupling().abs().flatten(1).amax(dim=1)
    assert torch.allclose(scaled, torch.ones(4), atol=1e-5)


def test_on_8x8_the_old_rule_left_most_kernels_weaker_than_the_ceiling():
    peaks = []
    for channels in plan.CHANNELS:
        for seed in SEEDS:
            peaks += physics_block(am.build_untrained(am.Arm("network", channels=channels, grid=8), 1.0, seed).core
                                   ).peaks().tolist()
    below = sum(p < 1.0 for p in peaks) / len(peaks)
    assert 0.5 < below < 1.0 and 0.7 < sorted(peaks)[len(peaks) // 2] < 0.9
    arm = am.Arm("network", grid=8)
    assert not torch.equal(_operator(arm, 0, "exact", "fft"), _operator(arm, 0, "cap", "fft"))


def test_the_factor_scales_up_as_well_as_down_and_the_cap_only_down():
    peak = torch.tensor([0.5, 1.0, 2.0])
    assert torch.allclose(ceiling_factor(peak, 1.0, "exact") * peak, torch.ones(3), atol=1e-6)
    assert ceiling_factor(peak, 1.0, "cap").tolist() == pytest.approx([1.0, 1.0, 0.5], abs=1e-6)
    with pytest.raises(ValueError, match="unknown kernel scaling"):
        ceiling_factor(peak, 1.0, "clip")

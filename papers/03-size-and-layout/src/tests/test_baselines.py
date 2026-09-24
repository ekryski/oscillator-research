"""The trained baselines: paper 02's at 2,048 parameters, and sized to every network of paper 03."""

import pytest
import torch

from harness.experiment import arms as am
from harness.experiment import plan

PAPER02 = {"gru": (18, 1944), "tcn": (12, 1948), "cnn": (13, 2109), "transformer": (16, 1968), "s4d": (16, 1840)}


@pytest.mark.parametrize("arch", am.ANNS)
def test_at_2048_parameters_on_16_rows_the_widths_are_paper_02s(arch):
    width, count = PAPER02[arch]
    assert am.ann_width(arch, 16, 2048) == width
    assert am.ann_params(arch, 16, width) == count
    torch.manual_seed(0)
    assert sum(p.numel() for p in am.build_ann(am.Arm("ann", arch=arch)).parameters()) == count


@pytest.mark.parametrize("arch", am.ANNS)
def test_every_width_is_the_widest_under_the_budget_and_the_cnn_the_narrowest_over_it(arch):
    step = 2 if arch == "transformer" else 1
    for grid, bands in plan.lattices():
        rows = grid
        for channels in plan.CHANNELS:
            budget = am.Arm("ann", arch=arch, channels=channels, grid=grid, bands=bands).budget
            w = am.ann_width(arch, rows, budget)
            if arch == "cnn":
                assert am.ann_params(arch, rows, w) >= budget and am.ann_params(arch, rows, w - step) < budget
            elif am.ann_params(arch, rows, w) <= budget:
                assert am.ann_params(arch, rows, w + step) > budget
            else:
                assert w == (2 if arch == "transformer" else 1)       # even the narrowest is over budget


def test_from_2048_parameters_up_every_baseline_is_within_15_percent_of_its_network():
    for grid, bands in plan.lattices():
        for channels in plan.CHANNELS:
            for arch in am.ANNS:
                arm = am.Arm("ann", arch=arch, channels=channels, grid=grid, bands=bands)
                if arm.budget >= 2048:
                    n = am.ann_params(arch, grid, am.ann_width(arch, grid, arm.budget))
                    assert abs(n / arm.budget - 1) <= am.BUDGET_TOLERANCE, (arm.label(), n)


@pytest.mark.parametrize("arch", am.ANNS)
def test_the_hidden_trajectory_is_causal(arch):
    torch.manual_seed(0)
    m = am.build_ann(am.Arm("ann", arch=arch, channels=1, grid=8))
    rows = torch.rand(2, 40, 8)
    r2 = rows.clone()
    r2[:, -1] += 1.0
    with torch.no_grad():
        h1, h2 = m._hidden(rows), m._hidden(r2)
    assert torch.allclose(h1[:, :-1], h2[:, :-1], atol=1e-5)

"""The trained baselines: paper 02's at 2,048 parameters, and sized to every network of paper 03."""

import pytest
import torch

from harness.experiment import arms as am
from harness.experiment import plan

PAPER02 = {"gru": (18, 1944), "tcn": (10, 1972), "cnn": (13, 2109), "transformer": (16, 1968), "s4d": (16, 1840)}


@pytest.mark.parametrize("arch", am.TRAINED)
def test_at_2048_parameters_on_16_rows_the_widths_are_paper_02s(arch):
    width, count = PAPER02[arch]
    assert am.trained_width(arch, 16, 2048) == width
    assert am.trained_params(arch, 16, width) == count
    torch.manual_seed(0)
    assert sum(p.numel() for p in am.build_trained(am.Arm("trained", arch=arch)).parameters()) == count


@pytest.mark.parametrize("arch", am.TRAINED)
def test_every_width_is_the_widest_under_the_budget_and_the_cnn_the_narrowest_over_it(arch):
    step = 2 if arch == "transformer" else 1
    for grid, bands in plan.lattices():
        rows = grid
        for channels in plan.CHANNELS:
            budget = am.Arm("trained", arch=arch, channels=channels, grid=grid, bands=bands).budget
            w = am.trained_width(arch, rows, budget)
            if arch == "cnn":
                assert am.trained_params(arch, rows, w) >= budget and am.trained_params(arch, rows, w - step) < budget
            elif am.trained_params(arch, rows, w) <= budget:
                assert am.trained_params(arch, rows, w + step) > budget
            else:
                assert w == (2 if arch == "transformer" else 1)       # even the narrowest is over budget


def test_from_2048_parameters_up_every_baseline_is_within_15_percent_of_its_network():
    for grid, bands in plan.lattices():
        for channels in plan.CHANNELS:
            for arch in am.TRAINED:
                arm = am.Arm("trained", arch=arch, channels=channels, grid=grid, bands=bands)
                if arm.budget >= 2048:
                    n = am.trained_params(arch, grid, am.trained_width(arch, grid, arm.budget))
                    assert abs(n / arm.budget - 1) <= am.BUDGET_TOLERANCE, (arm.label(), n)


@pytest.mark.parametrize("arch", am.TRAINED)
def test_the_hidden_trajectory_is_causal(arch):
    torch.manual_seed(0)
    m = am.build_trained(am.Arm("trained", arch=arch, channels=1, grid=8))
    rows = torch.rand(2, 40, 8)
    r2 = rows.clone()
    r2[:, -1] += 1.0
    with torch.no_grad():
        h1, h2 = m._hidden(rows), m._hidden(r2)
    assert torch.allclose(h1[:, :-1], h2[:, :-1], atol=1e-5)


def test_the_tcn_is_a_dilated_residual_tcn_and_the_cnn_sees_nine_frames():
    torch.manual_seed(0)
    tcn, cnn = am.build_trained(am.Arm("trained", arch="tcn")), am.build_trained(am.Arm("trained", arch="cnn"))
    for model, field in ((tcn, 31), (cnn, 9)):
        rows = torch.rand(1, 60, 16)
        hit = rows.clone()
        hit[:, 20] += 1.0                                  # a frame's influence reaches `field` frames, no further
        with torch.no_grad():
            moved = (model._hidden(hit) - model._hidden(rows)).abs().amax(dim=2)[0]
        reached = (moved > 1e-6).nonzero().flatten() + am.WARMUP_FRAMES
        assert reached.min().item() == 20 and reached.max().item() == 20 + field - 1, type(model).__name__
    zero = am.build_trained(am.Arm("trained", arch="tcn"))
    with torch.no_grad():
        for block in zero.blocks:
            for conv in block:
                conv.weight.zero_()
                conv.bias.zero_()
    rows = torch.rand(2, 40, 16)
    assert torch.equal(zero._hidden(rows), rows[:, am.WARMUP_FRAMES:])     # the residual path passes the input


def test_the_s4ds_time_scales_span_2_to_200_frames_and_its_output_is_linear():
    torch.manual_seed(0)
    s4d = am.build_trained(am.Arm("trained", arch="s4d"))
    rates = torch.exp(s4d.log_decay.detach())[0]
    assert rates[0].item() == pytest.approx(0.5) and rates[-1].item() == pytest.approx(0.005)
    assert torch.all(rates[1:] < rates[:-1])
    with torch.no_grad():
        assert (s4d._hidden(torch.rand(2, 40, 16)) < 0).any()           # no rectifier on the way out


def test_the_trained_head_reads_the_statistics_standardized():
    rows, tvalid = torch.rand(64, 30, 8), torch.full((64,), 30)
    labels = torch.arange(64) % 10
    _, head, _ = am.train_baseline(am.Arm("trained", arch="gru", channels=1, grid=8), rows, tvalid, labels,
                                   "recognition", 0, epochs=1)
    assert isinstance(head[0], torch.nn.BatchNorm1d) and not head[0].affine and isinstance(head[1], torch.nn.Linear)

"""Every part of a run works on the CPU, on Apple Silicon's GPU (MPS) and on CUDA.

The CPU is the reference: the only device on which paper 03 is bit-identical
to paper 02. On a GPU the results are close to the CPU's, not identical: its
FFTs, matrix products and reductions sum in other orders. The GPU tests skip
where the device is absent; this project's Mac (M1 Max) has MPS.
"""

import math

import pytest
import torch

from harness.confirm import arms as am
from harness.confirm import readout as ro
from harness.confirm import run as rn
from harness.confirm import stream as st
from harness.models.geometries import BOUNDARIES
from harness.stimuli.filterbank import bandpass_rows
from harness.stimuli.frontend import hop_rows, hop_rows_quad
from harness.utils import device as dv

from .test_confirm_run import bank, spec  # noqa: F401  (the synthetic bank fixture)

GPUS = [d for d in ("mps", "cuda") if dv.available(d)]
gpu = pytest.mark.parametrize("device", GPUS or [pytest.param("none", marks=pytest.mark.skip("no GPU here"))])


def test_auto_picks_cuda_then_mps_then_the_cpu():
    expected = "cuda" if dv.available("cuda") else "mps" if dv.available("mps") else "cpu"
    assert dv.resolve("auto") == expected and dv.resolve("cpu") == "cpu"
    assert dv.describe("cpu") == {"device": "cpu"}
    with pytest.raises(ValueError, match="unknown device"):
        dv.resolve("tpu")


def test_a_missing_gpu_is_an_error_not_a_silent_cpu_run():
    for d in ("mps", "cuda"):
        if not dv.available(d):
            with pytest.raises(RuntimeError, match="not available"):
                dv.resolve(d)


def _close(a: torch.Tensor, b: torch.Tensor, atol: float) -> None:
    assert a.device.type == "cpu" and b.device.type == "cpu"
    assert torch.allclose(a, b, atol=atol), (a - b).abs().max().item()


@gpu
@pytest.mark.parametrize("physics", ["kuramoto", "sakaguchi", "harmonic2", "winfree", "sl", "sl-fixedamp"])
def test_every_coupling_function_and_geometry_runs_on_the_gpu_as_on_the_cpu(device, physics):
    shapes = ("torus",) if physics.startswith("sl") else BOUNDARIES
    rows = torch.rand(2, 24, 8, generator=torch.Generator().manual_seed(1)) * 1.5
    for shape in shapes:
        for grid in (8, 16):
            arm = am.Arm("field", physics=physics, boundary=shape, channels=2, grid=grid, bands=grid)
            r = rows if grid == 8 else rows.repeat_interleave(2, dim=-1)
            with torch.no_grad():
                cpu = am.frozen_signals(arm, am.build_frozen(arm, 1.0, 0), r)
                gpu_ = am.frozen_signals(arm, am.build_frozen(arm, 1.0, 0, device), r.to(device)).cpu()
            _close(gpu_, cpu, atol=2e-3)


@gpu
def test_the_large_lattices_and_the_quadrature_drive_run_on_the_gpu(device):
    for grid in (32, 64):
        arm = am.Arm("field", channels=1, grid=grid, bands=grid)
        rows = torch.rand(1, 12, grid, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            cpu = am.frozen_signals(arm, am.build_frozen(arm, 1.0, 0), rows)
            gpu_ = am.frozen_signals(arm, am.build_frozen(arm, 1.0, 0, device), rows.to(device)).cpu()
        _close(gpu_, cpu, atol=2e-3)
    arm = am.Arm("field", channels=2, grid=8, bands=8)
    quad = torch.randn(2, 12, 8, 2, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        cpu = am.frozen_signals(arm, am.build_frozen(arm, 1.0, 0), quad)
        gpu_ = am.frozen_signals(arm, am.build_frozen(arm, 1.0, 0, device), quad.to(device)).cpu()
    _close(gpu_, cpu, atol=2e-3)


@gpu
def test_the_bank_and_the_front_ends_run_on_the_gpu(device):
    arm = am.Arm("bank", channels=2, grid=8, bands=8)
    rows = torch.rand(2, 30, 8, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        _close(am.frozen_signals(arm, am.build_frozen(arm, 1.0, 0, device), rows.to(device)).cpu(),
               am.frozen_signals(arm, am.build_frozen(arm, 1.0, 0), rows), atol=1e-5)
    waves = torch.randn(2, 16000, generator=torch.Generator().manual_seed(5)) * 0.2
    for bands in (16, 128):
        _close(hop_rows(waves.to(device), bands).cpu(), hop_rows(waves, bands), atol=1e-3)
        _close(hop_rows_quad(waves.to(device), bands).cpu(), hop_rows_quad(waves, bands), atol=2e-3)
    _close(bandpass_rows(waves.to(device), 16).cpu(), bandpass_rows(waves, 16), atol=1e-4)


@gpu
@pytest.mark.parametrize("arch", list(am.ANNS))
def test_every_trained_baseline_starts_from_the_cpus_weights_and_trains_on_the_gpu(device, arch, monkeypatch):
    arm = am.Arm("ann", arch=arch, channels=1, grid=8)
    rows, tvalid = torch.rand(40, 30, 8, generator=torch.Generator().manual_seed(6)), torch.full((40,), 30)
    labels = torch.arange(40) % 10
    starts, build = [], am.build_ann

    def recording(a):
        model = build(a)
        starts.append({k: v.clone() for k, v in model.state_dict().items()})
        return model
    monkeypatch.setattr(am, "build_ann", recording)
    am.train_ann(arm, rows, tvalid, labels, "recognition", 0, epochs=1)
    backbone, _, health = am.train_ann(arm, rows, tvalid, labels, "recognition", 0, epochs=1, device=device)
    assert all(torch.equal(starts[0][k], starts[1][k]) for k in starts[0]), "the same starting weights"
    assert next(backbone.parameters()).device.type == torch.device(device).type
    assert math.isfinite(health["last_epoch_loss"])


@gpu
def test_the_projection_runs_on_the_gpu_and_the_ridge_stays_on_the_cpu(device):
    x = torch.randn(300, 2000, generator=torch.Generator().manual_seed(7))
    mean, sd = ro._stats(x[:200])
    _close(ro._projected([x], slice(0, 300), mean, sd, 256, device=device),
           ro._projected([x], slice(0, 300), mean, sd, 256), atol=1e-3)


@gpu
@pytest.mark.parametrize("streamed", [False, True], ids=["in memory", "streamed"])
def test_a_whole_run_on_the_gpu_is_close_to_the_same_run_on_the_cpu(bank, device, streamed, monkeypatch):  # noqa: F811
    monkeypatch.setattr(st, "STREAM_STATES", 0 if streamed else 10**9)
    s = spec(am.Arm("field", channels=2, grid=8), sizes=(128,), widths=(64, 256), native_sizes=(),
             reads=("windowed",))
    cpu, gpu_ = rn.execute(s, "cpu", bank=bank), rn.execute(s, device, bank=bank)
    assert gpu_["env"]["device"] == device and "gpu" in gpu_["env"] and cpu["env"]["device"] == "cpu"
    mine = {(c["width"], c["projection"]): c["acc"] for c in cpu["cells"]}
    for c in gpu_["cells"]:
        assert abs(c["acc"] - mine[(c["width"], c["projection"])]) <= 0.05


@gpu
def test_a_trained_baseline_runs_whole_on_the_gpu(bank, device):  # noqa: F811
    rec = rn.execute(spec(am.Arm("ann", arch="gru", channels=1, grid=8), sizes=(128,), native_sizes=(128,)),
                     device, bank=bank)
    assert rec["env"]["device"] == device and all(0.0 <= c["acc"] <= 1.0 for c in rec["cells"])

"""The GPU benchmark: what it times, and how it extrapolates a run from it."""

import json
import math

import pytest
import torch

from harness.confirm import benchmark as bm
from harness.confirm import plan
from harness.confirm import run as rn
from harness.confirm.arms import Arm

FAST = {"matmul_flops": 1e11, "randn_per_s": 5e7, "ridge_s": 10.0}


def test_a_small_benchmark_times_both_pathways_and_writes_every_tier(tmp_path):
    out = tmp_path / "bench.json"
    report = bm.benchmark("cpu", grids=(8,), channels=(1,), designs=False, max_clips=2, out=out,
                          log=lambda *_: None, throughput=FAST)
    saved = json.loads(out.read_text())
    assert saved["totals"] == json.loads(json.dumps(report["totals"]))
    assert {(c["pathway"], c["kind"]) for c in saved["cells"]} == {
        ("envelope", "field"), ("envelope", "bank"), ("carrier", "field"), ("carrier", "bank")}
    assert all(c["seconds_per_clip"] > 0 and c["clips_timed"] == 2 for c in saved["cells"])
    assert set(saved["front_end_s_per_clip"]) == {"8", "16"}
    assert set(saved["tiers"]) == set(plan.TIERS)
    runs = sum(len(plan.planned([t])) - sum(map(plan.reused, plan.planned([t]))) for t in plan.TIERS)
    assert len(saved["runs"]) == runs
    measured = {r["run"] for r in saved["runs"] if r["measured"]}
    assert "carrier-carrier-8x8/A/0db/g32/s0/field-kuramoto-torus-random-lam0.3-clamp1-c1-8x8" in measured
    assert not any("16x16" in r for r in measured), "only the timed lattice counts as measured"


def test_a_streamed_arm_is_timed_one_channel_at_a_time():
    cell = bm.time_arm(Arm("field", channels=8, grid=32), "envelope", "cpu", max_clips=2)
    assert cell["streamed"] and cell["channels_timed"] == 1


@pytest.mark.parametrize("drive", ["envelope", "carrier"])
def test_a_run_is_extrapolated_from_its_clips_frames_reads_and_passes(drive):
    arm = Arm("field", grid=8, channels=1)
    per_clip = 1e-3
    m = {"cells": {(drive, 8, 1, "field"): {"seconds_per_clip": per_clip}}, "designs": {},
         "front_end": {8: 0.0, 16: 0.0}, "throughput": {"matmul_flops": math.inf, "randn_per_s": math.inf,
                                                         "ridge_s": 0.0}}
    gain = plan.CARRIER_GAIN if drive == "carrier" else 1.0
    spec = plan._net("size", drive, 0.0, gain, 0, arm)
    secs, measured = bm.run_seconds(spec, m)
    assert measured and secs == pytest.approx(plan.CLIPS * per_clip)
    if drive == "envelope":
        seq = plan._net("sequence", drive, 0.0, 1.0, 0, arm, "sequence", length=4)
        clips, frames = plan.clips_and_frames(seq)
        assert bm.run_seconds(seq, m)[0] == pytest.approx(clips * per_clip * frames / 61)
        m["designs"][(drive, 8, "winfree", "cube")] = 2.0
        cube = plan._net("design", drive, 0.0, 1.0, 0, Arm("field", grid=8, channels=1, physics="winfree", boundary="cube"))
        assert bm.run_seconds(cube, m)[0] == pytest.approx(2 * plan.CLIPS * per_clip)


def test_the_baselines_and_untimed_arms_are_modelled():
    m = {"cells": {}, "designs": {}, "front_end": {}, "throughput": FAST}
    floor = rn.Spec("size", "recognition", "envelope", 0.0, None, 0, Arm("floor", grid=8))
    assert bm.run_seconds(floor, m) == (plan.seconds(floor, "mps"), False)
    net = plan._net("size", "envelope", 0.0, 1.0, 0, Arm("field", grid=64))
    assert bm.run_seconds(net, m) == (plan.seconds(net, "mps"), False)


def test_a_batch_too_large_for_the_device_is_halved_until_it_fits(monkeypatch):
    real = bm.am.frozen_signals

    def small_device(arm, model, rows):
        if len(rows) > 1:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(arm, model, rows)
    monkeypatch.setattr(bm.am, "frozen_signals", small_device)
    cell = bm.time_arm(Arm("field", channels=1, grid=8), "envelope", "cpu", max_clips=4)
    assert cell["clips_timed"] == 1 and not cell["fits"] and cell["seconds_per_clip"] > 0
    monkeypatch.setattr(bm.am, "frozen_signals", lambda *a: (_ for _ in ()).throw(RuntimeError("out of memory")))
    cell = bm.time_arm(Arm("field", channels=1, grid=8), "envelope", "cpu", max_clips=4)
    assert cell["seconds_per_clip"] is None
    m = {"cells": {("envelope", 8, 1, "field"): cell}, "designs": {}, "front_end": {}, "throughput": FAST}
    net = plan._net("size", "envelope", 0.0, 1.0, 0, Arm("field", channels=1, grid=8))
    assert bm.run_seconds(net, m) == (plan.seconds(net, "mps"), False)

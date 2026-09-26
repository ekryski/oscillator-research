"""Paper 03's tiers: what they run, what they take from paper 02, and what they cost."""

import pytest

from harness.experiment import plan
from harness.experiment import run as rn
from harness.experiment.arms import Arm

#: the run counts TIERS.md and DESIGN.md state
COUNTS = {"gate": 30, "size": 2688, "trained": 825, "sequence": 1296, "design": 3375, "quadrature": 363,
          "design-quadrature": 3105}
REUSED = {"gate": 0, "size": 90, "trained": 15, "sequence": 0, "design": 75, "quadrature": 6,
          "design-quadrature": 12}


@pytest.mark.parametrize("tier", plan.TIERS)
def test_each_tier_has_the_documented_runs_and_no_two_share_an_address(tier):
    specs = plan.planned([tier])
    assert len(specs) == COUNTS[tier]
    assert sum(map(plan.reused, specs)) == REUSED[tier]


def test_every_tier_runs_at_0_db_and_input_gain_1_on_the_spectrogram_and_quadrature_pathways():
    specs = plan.planned(list(plan.TIERS))
    assert {s.noise_db for s in specs} == {0.0}
    assert {s.gain for s in specs if s.tier != "gate"} == {None, 1.0}
    assert {s.pathway for s in specs} == {"spectrogram", "quadrature"}


def test_the_lattices_are_nine_and_16x16_runs_once():
    assert list(plan.lattices()) == [(8, 0), (8, 16), (16, 0), (32, 0), (32, 16), (64, 0), (64, 16), (128, 0),
                                     (128, 16)]


def test_the_long_window_runs_only_at_64_and_128_with_one_band_per_row():
    assert [f for f in plan.front_ends() if f[2]] == [(64, 0, 1024), (128, 0, 2048)]
    specs = plan.planned(["size", "trained", "quadrature"])
    assert {(s.arm.grid, s.arm.n_bands, s.arm.n_window) for s in specs if s.arm.window} == {
        (64, 64, 1024), (128, 128, 2048)}
    assert all(not s.arm.window for s in plan.planned(["sequence", "design", "design-quadrature"]))


def test_the_size_tier_carries_paper_02s_order_task_and_the_sequence_tier_every_length():
    order = [s for s in plan.planned(["size"]) if s.task == "order"]
    assert {s.pair for s in order} == set(plan.pr.PAIRS)
    assert sum(map(plan.reused, order)) == 75                           # paper 02's tier 1 order runs
    seq = plan.planned(["sequence"])
    assert {s.length for s in seq} == set(plan.pr.SEQUENCE_LENGTHS) and {s.task for s in seq} == {"sequence"}
    assert not any(s.arm.kind == "trained" for s in seq)


def test_a_network_records_its_rotation_rates_and_the_baseline_its_whole_clip_read():
    assert plan.reads_for(Arm("network"), "recognition") == ("windowed", "windowed+rate")
    assert plan.reads_for(Arm("network"), "sequence") == ("pooled", "pooled+rate")
    assert plan.reads_for(Arm("bank"), "order") == ("pooled",)
    assert plan.reads_for(Arm("baseline"), "recognition") == ("windowed", "windowed@wholeclip")


def test_the_reused_cells_are_paper_02s_16x16_4_channel_runs_under_their_own_ids():
    s = rn.Spec("size", "recognition", "spectrogram", 0.0, 1.0, 2, Arm("network"))
    assert plan.paper02_group(s) == "tier1-recognition-spectrogram"
    assert s.run_id() == "A/0db/g1/s2/coupled-kuramoto-torus-random-restoring0.3-ceiling1"
    wide = rn.Spec("size", "recognition", "spectrogram", 0.0, 1.0, 2, Arm("bank", channels=8))
    assert plan.paper02_group(wide) == "tier1-recognition-spectrogram"      # paper 02's width-matched bank
    assert wide.run_id() == "A/0db/g1/s2/bank-width"
    d = rn.Spec("design", "recognition", "spectrogram", 5.0, 2.0, 0, Arm("network", coupling="second-harmonic", geometry="helix"))
    assert plan.paper02_group(d) == "tier2-recognition-spectrogram-second-harmonic"
    for arm in (Arm("network", channels=8), Arm("network", grid=32), Arm("network", coupled=False, coupling="winfree"),
                Arm("bank", channels=2), Arm("trained", arch="gru", channels=8)):
        assert plan.paper02_group(rn.Spec("size", "recognition", "spectrogram", 0.0, 1.0, 0, arm)) is None
    q = rn.Spec("design-quadrature", "recognition", "quadrature", 0.0, 1.0, 0, Arm("network", coupling="winfree"))
    assert plan.paper02_group(q) == "tier3-recognition-quadrature"
    assert plan.paper02_group(rn.Spec("design-quadrature", "recognition", "quadrature", 0.0, 1.0, 0,
                                      Arm("network", coupling="winfree", geometry="cube"))) is None


def test_the_largest_arms_are_streamed_and_everything_paper_02_ran_is_not():
    specs = plan.planned(["size"])
    assert not any(s.streamed for s in specs if plan.reused(s))
    assert all(s.streamed for s in specs if s.arm.kind != "baseline" and s.arm.states > 4096)


def test_a_stage_selects_lattices_and_channels_without_changing_a_spec():
    small = plan.planned(["size"], grids=(8, 16, 32))
    assert {s.arm.grid for s in small} == {8, 16, 32}
    full = {(s.group(), s.run_id()) for s in plan.planned(["size"])}
    assert {(s.group(), s.run_id()) for s in small} <= full


def test_the_estimate_covers_every_tier_and_lattice():
    rows = plan.estimate()
    assert {r["tier"] for r in rows} == set(plan.TIERS)
    assert sum(r["runs"] for r in rows) == sum(COUNTS.values())
    assert all(r["cpu_hours"] >= 0 and r["peak_gb"] > 0 for r in rows)


def _cell(read, width, effective, acc, projection=None):
    c = {"read": read, "n_train": 2048, "width": width, "effective_width": effective, "acc": acc}
    return c if projection is None else {**c, "projection": projection}


def test_paper_02s_untagged_cells_and_its_projection_tier_are_read_together(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(plan, "PAPER02_RECORD", tmp_path)
    s = rn.Spec("size", "recognition", "spectrogram", 0.0, 1.0, 0, Arm("network"), reads=("windowed",))
    old = {"native_widths": {"windowed": 24576, "pooled": 6144},
           "cells": [_cell("windowed", 192, 192, 0.78), _cell("windowed", "native", 24576, 0.81),
                     _cell("pooled", 192, 192, 0.70)]}
    (tmp_path / "tier1-recognition-spectrogram.json").write_text(json.dumps({"runs": {s.run_id(): old}}))
    rec = plan.paper02_run(s)
    assert [c["projection"] for c in rec["cells"]] == ["fixed", "none", "fixed"]
    assert not plan.paper02_complete(s, rec), "paper 02's tier 1 alone has no seeded projection"
    again = {"cells": [_cell("windowed", 192, 192, 0.78, "fixed"), _cell("windowed", 192, 192, 0.77, "seeded"),
                       _cell("windowed", "native", 24576, 0.81, "none")]}
    (tmp_path / "projection-recognition-spectrogram.json").write_text(json.dumps({"runs": {s.run_id(): again}}))
    rec = plan.paper02_run(s)
    assert sorted((c["read"], str(c["width"]), c["projection"]) for c in rec["cells"]) == [
        ("pooled", "192", "fixed"), ("windowed", "192", "fixed"), ("windowed", "192", "seeded"),
        ("windowed", "native", "none")]
    assert plan.paper02_complete(s, rec)
    assert plan.pending([s]) == []


def test_a_paper_02_run_with_no_projected_read_needs_no_seeded_cell(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(plan, "PAPER02_RECORD", tmp_path)
    s = rn.Spec("size", "recognition", "spectrogram", 0.0, None, 0, Arm("baseline"), reads=("windowed",))
    old = {"native_widths": {"windowed": 192}, "cells": [_cell("windowed", 192, 192, 0.78)]}
    (tmp_path / "tier1-recognition-spectrogram.json").write_text(json.dumps({"runs": {s.run_id(): old}}))
    assert plan.paper02_complete(s, plan.paper02_run(s))


def test_prepare_builds_every_row_cache_a_run_reads_and_no_other():
    paths = {plan.cache_path(j) for j in plan.cache_jobs()}
    wanted = set()
    for s in plan.planned([t for t in plan.TIERS if t != "gate"]):
        a = s.arm
        if s.task == "recognition":
            wanted.add(plan.pr.rows_path(s.pathway, s.noise_db, a.n_bands, a.n_window))
        else:
            for code in (0, s.seed + 1):
                wanted.add(plan.pr.order_rows_path(s.pair, code, s.noise_db, a.n_bands) if s.task == "order"
                           else plan.pr.sequence_rows_path(s.length, code, s.noise_db, a.n_bands))
    assert wanted == paths


def test_paper_02s_own_record_is_read_under_its_labels_where_it_is_present():
    """Against paper 02's record itself (papers/02-untrained-reservoirs/results/, or OSC_PAPER02_RESULTS)."""
    if not (plan.PAPER02_RECORD / "tier1-recognition-spectrogram.json").exists():
        pytest.skip("paper 02's renamed record is not here; set OSC_PAPER02_RESULTS to its results/")
    for arm in (Arm("network"), Arm("network", coupled=False), Arm("bank", channels=8), Arm("baseline")):
        s = rn.Spec("size", "recognition", "spectrogram", 0.0, None if arm.kind == "baseline" else 1.0, 0, arm)
        rec = plan.paper02_run(s)
        assert rec is not None and rec["spec"]["pathway"] == "spectrogram" and rec["spec"]["arm"]["kind"] == arm.kind
        assert {c["projection"] for c in rec["cells"]} <= {"fixed", "seeded", "none"}

"""The summary on a small hand-built record whose answers are known, and paper 02's cells folded in."""

import json

import numpy as np
import pytest
import torch

from harness.experiment import plan
from harness.experiment import readout as ro
from harness.experiment import summary as sm
from harness.experiment.arms import Arm

N_TEST = 400
FIELD_LABEL = "coupled-kuramoto-torus-random-restoring0.3-ceiling1"


def bits(acc: float, seed: int) -> str:
    correct = np.zeros(N_TEST, dtype=bool)
    correct[np.random.default_rng(seed).choice(N_TEST, int(round(acc * N_TEST)), replace=False)] = True
    return ro.pack(torch.from_numpy(correct))


def run(arm: Arm, noise, gain, seed, acc, read, experiment="size"):
    spec = {"experiment": experiment, "task": "recognition", "pathway": "spectrogram", "noise_db": noise, "gain": gain,
            "seed": seed, "arm": arm.as_dict(), "span": "fixed"}
    # recorded as paper 02 records: no projection tag, so the summary reads them as the fixed projection
    cells = [{"read": read, "width": 192, "effective_width": 192, "n_train": 2048, "acc": acc,
              "correct": bits(acc, 7 * seed + 2)}]
    return {"spec": spec, "cells": cells, "n_test": N_TEST, "native_widths": {read: 10**6}}


def paper02_record(root, **files) -> None:
    """A paper 02 record at `root`: every file paper 03 reads, empty unless given."""
    root.mkdir(exist_ok=True)
    groups = {plan.paper02_group(s) for e in plan.EXPERIMENTS.values() for s in e()} - {None}
    for group in groups | {f"{plan.PROJECTION02}-{task}" for task in ("recognition", "order")}:
        (root / f"{group}.json").write_text(json.dumps({"runs": files.get(group, {})}))


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(plan, "PAPER02_RECORD", tmp_path / "paper02")
    runs, old = {}, {}
    for noise in (0.0, 5.0):
        for seed in (0, 1, 2):
            runs[f"A/{noise:g}db/s{seed}/baseline-32x32"] = run(Arm("baseline", grid=32), noise, None, seed, 0.70,
                                                              "windowed@wholeclip")
            for gain in (1.0, 2.0):
                net = Arm("network", channels=8, grid=32)
                runs[f"A/{noise:g}db/g{gain:g}/s{seed}/{net.label()}"] = run(net, noise, gain, seed,
                                                                             0.78 + 0.01 * seed, "windowed")
                # paper 02's 16 x 16, 4-channel network, recorded in its own controls experiment's file
                old[f"A/{noise:g}db/g{gain:g}/s{seed}/{FIELD_LABEL}"] = run(Arm("network"), noise, gain, seed, 0.75,
                                                                           "windowed", experiment="controls")
    (tmp_path / "size-recognition-32x32.json").write_text(json.dumps({"runs": runs}))
    paper02_record(tmp_path / "paper02", **{"controls-recognition": old})
    return tmp_path


def test_an_accuracy_is_its_mean_spread_and_every_seed_in_points():
    s = sm.spread({"seed0": 0.78, "seed1": 0.79, "seed2": 0.80})
    assert s["mean"] == pytest.approx(79.0) and s["sd"] == pytest.approx(1.0) and s["n"] == 3


def test_the_network_is_compared_with_its_own_lattices_baseline_at_every_gain(recorded):
    s = sm.summary()
    rows = [r for r in s["comparisons"] if r["comparison"].endswith("spectrogram-only baseline, whole clip")
            and r["comparison"].startswith("coupled")]
    assert {(r["grid"], r["channels"], r["gain"], r["noise"]) for r in rows} == {
        (32, 8, g, n) for g in (1.0, 2.0) for n in (0.0, 5.0)}
    for r in rows:
        assert r["mean"] == pytest.approx(9.0) and r["n"] == 3 and len(r["ci95"]) == 2


def test_paper_02s_cells_are_read_from_its_record_under_paper_03s_experiment(recorded):
    acc = sm.summary()["accuracy"]
    reused = [r for r in acc if r["arm"] == FIELD_LABEL]
    assert reused and all(r["experiment"] == "size" and r["source"] == ["paper 02"] for r in reused)
    assert all(r["mean"] == pytest.approx(75.0) for r in reused)


def test_a_cell_without_a_projection_tag_is_paper_02s_fixed_one_or_unprojected():
    rec = {"native_widths": {"windowed": 192}}
    assert sm.projection_of({"effective_width": 192, "read": "windowed"}, rec) == "none"
    assert sm.projection_of({"effective_width": 64, "read": "windowed"}, rec) == "fixed"
    assert sm.projection_of({"projection": "seeded", "effective_width": 64, "read": "windowed"}, rec) == "seeded"


def test_the_report_tabulates_lattices_against_channels(recorded):
    s = sm.summary()
    text = sm.report(s, sm.progress())
    assert "32 × 32 lattice, 32 mel bands" in text and "8 channels" in text and "(+" in text
    assert "fixed projection" in text and "seeded projection" in text


def _memory_run(arm: Arm, seed: int, task: str, cells: list[dict], **extra) -> dict:
    spec = {"experiment": "size" if task == "order" else "sequence", "task": task, "pathway": "spectrogram", "noise_db": 0.0,
            "gain": None if arm.kind == "baseline" else 1.0, "seed": seed, "arm": arm.as_dict(), "span": "fixed",
            **extra}
    return {"spec": spec, "cells": cells, "n_test": N_TEST, "native_widths": {"pooled": 10**6}}


def _cell(acc, seed, read="pooled", **extra):
    return {"read": read, "width": 192, "effective_width": 192, "n_train": 2048, "acc": acc, "projection": "fixed",
            "correct": bits(acc, seed), **extra}


def test_the_order_tasks_pairs_are_pooled_and_a_sequences_positions_taken_together(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(plan, "PAPER02_RECORD", tmp_path / "paper02")
    paper02_record(tmp_path / "paper02")
    net = Arm("network", channels=2, grid=8)
    order, seq = {}, {}
    for seed in (0, 1, 2):
        for i, pair in enumerate(plan.pr.PAIRS):
            order[f"A/pair{pair[0]}{pair[1]}/0db/g1/s{seed}/{net.label()}"] = _memory_run(
                net, seed, "order", [_cell(0.6 + 0.02 * i, 11 * i + seed)], pair=list(pair))
        cells = [_cell(0.5, 3 * seed + p, position=p) for p in range(3)]
        seq[f"A/seq3/0db/g1/s{seed}/{net.label()}"] = {**_memory_run(net, seed, "sequence", cells, length=3),
                                                       "instruments": {"R": {"mean": 0.2 + 0.1 * seed}}}
    root = tmp_path
    (root / "size-order-8x8.json").write_text(json.dumps({"runs": order}))
    (root / "sequence-sequence-8x8.json").write_text(json.dumps({"runs": seq}))
    s = sm.summary()
    pooled = [r for r in s["accuracy"] if r["task"] == "order" and r["pair"] == "all"]
    assert len(pooled) == 1 and pooled[0]["mean"] == pytest.approx(64.0)
    mean = [r for r in s["accuracy"] if r["task"] == "sequence" and r["position"] == "mean"]
    every = [r for r in s["accuracy"] if r["task"] == "sequence" and r["position"] == "all"]
    assert mean[0]["mean"] == pytest.approx(50.0) and every[0]["mean"] < 50.0
    inst = [r for r in s["instruments"] if r["task"] == "sequence"]
    assert inst[0]["R"]["mean"] == pytest.approx(0.3) and inst[0]["R"]["n"] == 3 and inst[0]["length"] == 3
    assert s["chance"]["sequence"][3]["order_free"] == pytest.approx(1 / 3)
    text = sm.report(s, sm.progress())
    assert "## Digit sequences of 3, every position right" in text and "Chance is 0.139%" in text
    assert "## Order task, the five pairs pooled" in text and "Chance is 50%" in text


def test_a_longer_window_is_compared_with_the_same_lattices_own_window_and_nothing_else():
    def cell(arm, seed, acc):
        return sm.Cell("size", "spectrogram", 0.0, 1.0, seed, arm.as_dict(), arm.label(), "windowed", 192, 2048, acc,
                       bits(acc, seed), N_TEST)
    long, short = Arm("network", grid=64, window=1024), Arm("network", grid=64)
    mapped = Arm("network", grid=64, bands=16)
    cells = [cell(a, s, acc) for s in (0, 1, 2) for a, acc in ((long, 0.8), (short, 0.7), (mapped, 0.6))]
    rows = [r for r in sm.size_comparisons(cells) if r["comparison"].endswith("the longer window minus paper 02's")]
    assert len(rows) == 1 and rows[0]["mean"] == pytest.approx(10.0) and rows[0]["window"] == 1024
    rows = [r for r in sm.size_comparisons(cells) if r["comparison"].endswith("16 bands mapped onto the rows")]
    assert len(rows) == 1 and rows[0]["mean"] == pytest.approx(10.0) and rows[0]["window"] == 0


def test_the_leak_check_lists_every_zero_input_cell_off_chance():
    def cell(arm, acc):
        return sm.Cell("leak-check", "spectrogram", 0.0, 0.0, 0, arm.as_dict(), arm.label(), "windowed", 192, 2048,
                       acc, None, N_TEST)
    big = Arm("network", channels=16, grid=128)
    lk = sm.leak_check([cell(Arm("network", grid=8), 0.1), cell(big, 0.1), cell(Arm("bank", grid=8), 0.1025)])
    assert lk["runs"] == 3 and lk["cells"] == 3 and len(lk["off_chance"]) == 1 and "bank" in lk["off_chance"][0]


def test_the_reuse_check_compares_every_cell_both_records_hold(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(plan, "PAPER02_RECORD", tmp_path / "paper02")
    first = next(plan.reuse_check())
    same, other = _cell(0.78, 1, "windowed"), _cell(0.61, 2, "windowed", width=1024)
    theirs = {"spec": {"experiment": "controls", "task": first.task, "pathway": first.pathway,
                       "noise_db": first.noise_db, "gain": first.gain, "seed": first.seed,
                       "arm": {k: v for k, v in first.arm.as_dict().items() if k not in ("grid", "bands", "window")}},
              "cells": [same, {**other, "acc": 0.60}], "native_widths": {"windowed": 24576}}
    paper02_record(tmp_path / "paper02", **{plan.paper02_group(first): {first.run_id(): theirs}})
    assert sm.reuse_check()["runs"] == 0                                   # nothing run here yet
    mine = {**theirs, "spec": first.as_dict(), "cells": [same, other, _cell(0.7, 3, "windowed+rate")]}
    (tmp_path / f"{first.group()}.json").write_text(json.dumps({"runs": {first.run_id(): mine}}))
    ru = sm.reuse_check()
    assert (ru["runs"], ru["cells"], ru["identical"], len(ru["differing"])) == (1, 2, 1, 1)
    assert len(ru["not_recorded"]) == len(list(plan.reuse_check())) - 1


def test_the_coil_is_compared_with_the_torus_over_the_phase_coupling_functions_at_each_size():
    def cell(experiment, coupling, geometry, seed, acc):
        arm = Arm("network", coupling=coupling, geometry=geometry, channels=2, grid=8)
        return sm.Cell(experiment, "spectrogram", 0.0, 1.0, seed, arm.as_dict(), arm.label(), "windowed", 192, 2048,
                       acc, bits(acc, seed), N_TEST)
    cells = []
    for seed in (0, 1, 2):
        cells += [cell("cochlea", "kuramoto", "coil", seed, 0.70), cell("size", "kuramoto", "torus", seed, 0.66),
                  cell("cochlea", "winfree", "coil", seed, 0.60), cell("design", "winfree", "torus", seed, 0.60),
                  cell("design", "winfree", "helix", seed, 0.50)]
    rows = {r["comparison"]: r for r in sm.cochlea_comparisons(cells)}
    coil = rows["lattice geometry: coil minus torus"]
    assert coil["n_pairs"] == 6 and coil["mean"] == pytest.approx(2.0) and (coil["grid"], coil["channels"]) == (8, 2)
    assert rows["lattice geometry: coil minus helix"]["mean"] == pytest.approx(10.0)     # Winfree's only
    assert "lattice geometry: cochlea minus coil" not in rows


def test_noise_is_reported_as_the_signal_to_noise_ratio():
    assert (sm.snr(0.0), sm.snr(5.0), sm.snr(None)) == ("0 dB", "−5 dB", "clean")


def test_a_quadrature_network_is_paired_with_the_same_network_on_the_spectrogram_pathway():
    def cell(experiment, pathway, coupling, geometry, seed, acc):
        arm = Arm("network", coupling=coupling, geometry=geometry, channels=1, grid=32)
        return sm.Cell(experiment, pathway, 0.0, 1.0, seed, arm.as_dict(), arm.label(), "windowed", 192, 2048, acc,
                       bits(acc, seed), N_TEST)
    cells = [c for seed in (0, 1, 2) for c in (
        cell("quadrature", "quadrature", "kuramoto", "torus", seed, 0.20), cell("size", "spectrogram", "kuramoto", "torus", seed, 0.70),
        cell("design-quadrature", "quadrature", "winfree", "cube", seed, 0.15), cell("design", "spectrogram", "winfree", "cube", seed, 0.65))]
    rows = {r["comparison"]: r for r in sm.pathway_comparisons(cells)}
    assert rows["coupled oscillator network: quadrature minus spectrogram pathway"]["mean"] == pytest.approx(-50.0)
    assert rows["Winfree, cube: quadrature minus spectrogram pathway"]["n_pairs"] == 3

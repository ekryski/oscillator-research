"""The raw-results summary on a small hand-built record whose answers are known."""

import json

import numpy as np
import pytest
import torch

from harness.experiment import readout as ro
from harness.experiment import record as rec
from harness.experiment import summary as sm

N_TEST = 400
COUPLED = {"kind": "network", "coupling": "kuramoto", "geometry": "torus", "frequencies": "random",
           "restoring": 0.3, "ceiling": 1.0, "channels": 4, "coupled": True, "arch": ""}
BASELINE = {**COUPLED, "kind": "baseline"}


def bits(acc: float, seed: int) -> str:
    correct = np.zeros(N_TEST, dtype=bool)
    correct[np.random.default_rng(seed).choice(N_TEST, int(round(acc * N_TEST)), replace=False)] = True
    return ro.pack(torch.from_numpy(correct))


def run(arm, noise, gain, seed, acc, read, experiment="controls", fold=-1):
    spec = {"experiment": experiment, "task": "recognition", "pathway": "spectrogram", "noise_db": noise, "gain": gain,
            "seed": seed, "arm": arm, "pair": [], "span": "fixed", "fold": fold}
    cells = [{"read": read, "width": 192, "n_train": 2048, "acc": acc, "correct": bits(acc, 7 * seed + fold + 2)}]
    return {"spec": spec, "cells": cells, "n_test": N_TEST}


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    runs = {}
    for noise in (0.0, 5.0):
        for seed in (0, 1, 2):
            runs[f"A/{noise:g}db/s{seed}/baseline"] = run(BASELINE, noise, None, seed, 0.70, "windowed@wholeclip")
            for gain in (1.0, 2.0):
                acc = 0.78 + 0.01 * seed
                runs[f"A/{noise:g}db/g{gain:g}/s{seed}/{rec.COUPLED_LABEL}"] = run(COUPLED, noise, gain, seed, acc,
                                                                                    "windowed")
    root = tmp_path
    (root / "controls-recognition.json").write_text(json.dumps({"runs": runs}))
    return root


def test_an_accuracy_is_its_mean_spread_and_every_replicate_in_points():
    s = sm.spread({"seed0": 0.78, "seed1": 0.79, "seed2": 0.80})
    assert s["mean"] == pytest.approx(79.0) and s["sd"] == pytest.approx(1.0) and s["n"] == 3
    assert s["values"] == pytest.approx({"seed0": 78.0, "seed1": 79.0, "seed2": 80.0})


def test_the_network_is_compared_with_the_gainless_baseline_at_every_gain(recorded):
    rows = [r for r in sm.controls(rec.load())
            if r["comparison"] == "coupled oscillator network minus the spectrogram-only baseline, whole clip"]
    assert {(r["noise"], r["gain"]) for r in rows} == {(0.0, 1.0), (0.0, 2.0), (5.0, 1.0), (5.0, 2.0)}
    for r in rows:
        assert r["mean"] == pytest.approx(9.0) and r["sd"] == pytest.approx(1.0)
        assert r["values"] == pytest.approx({"seed0": 8.0, "seed1": 9.0, "seed2": 10.0})
        assert r["ci95"][0] < 9.0 < r["ci95"][1]


def test_folds_are_replicates_and_their_test_clips_are_pooled_not_averaged():
    a = [rec.Cell("becker-folds", "recognition", "spectrogram", None, 1.0, 0, COUPLED, "f", (), "windowed", 192, 18000,
                 acc, bits(acc, f), N_TEST, f) for f, acc in enumerate((0.80, 0.90))]
    b = [rec.Cell("becker-folds", "recognition", "spectrogram", None, None, 0, BASELINE, "baseline", (), "windowed", 192, 18000,
                 0.75, bits(0.75, 10 + f), N_TEST, f) for f in range(2)]
    r = sm.paired(list(zip(a, b, strict=True)))
    assert r["values"] == pytest.approx({"fold0": 5.0, "fold1": 15.0}) and r["mean"] == pytest.approx(10.0)
    assert r["ci95"][0] < 10.0 < r["ci95"][1]


def test_the_summary_runs_before_every_experiment_is_in_and_the_record_ignores_it(recorded):
    sm.main([])
    text = (recorded / "summary.md").read_text()
    assert "(not run yet)" in text and "+9.00 ± 1.00" in text
    assert len(rec.load()) == 18                  # summary.json sits beside the record but is not read as one


def test_selection_picks_by_validation_accuracy_and_reads_the_test_accuracy_off_afterwards(tmp_path, monkeypatch):
    from harness.experiment import selection
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    runs = {}
    for seed in (0, 1, 2):
        for gain, restoring, val, test in ((1.0, 0.3, 0.70, 0.80), (0.5, 0.1, 0.75, 0.74), (2.0, 0.5, 0.60, 0.90)):
            arm = {**COUPLED, "restoring": restoring}
            label = f"coupled-kuramoto-torus-random-restoring{restoring:g}-ceiling1"
            r = run(arm, 0.0, gain, seed, test + 0.01 * seed, "windowed", experiment="selection")
            r["spec"]["protocol"] = "V"
            r["cells"][0]["val_acc"] = val + 0.01 * seed
            runs[f"V/0db/g{gain:g}/s{seed}/{label}"] = r
    (tmp_path / "selection-recognition-kuramoto.json").write_text(json.dumps({"runs": runs}))
    cells = rec.load()
    chosen = selection.select(cells, "selection", "kuramoto")
    assert set(chosen) == {0.0}
    pick = chosen[0.0]
    assert pick["gain"] == 0.5 and pick["label"].endswith("restoring0.1-ceiling1") and pick["grid"] == 3
    rows = sm.selections(cells)
    assert len(rows) == 1 and rows[0]["family"] == "kuramoto"
    assert rows[0]["selected"]["validation"]["mean"] == pytest.approx(76.0)
    assert rows[0]["selected"]["test"]["mean"] == pytest.approx(75.0)      # not the 90% the test set would have picked
    assert rows[0]["reference"]["test"]["mean"] == pytest.approx(81.0)
    assert rows[0]["selected_minus_reference"]["mean"] == pytest.approx(-6.0)
    axes = {(p["axis"], tuple(pt["value"] for pt in p["points"])) for p in sm.selection_profiles(cells)}
    assert axes == {("gain", (1.0,)), ("restoring", (0.3,)), ("ceiling", (1.0,))}   # only the reference sits on every axis


def test_the_calibration_reads_the_baseline_over_the_whole_clip_and_flags_the_window(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    runs = {}
    for noise, base, net in ((-10.0, 0.95, 0.93), (0.0, 0.80, 0.78), (10.0, 0.55, 0.60)):
        for seed in (0, 1, 2):
            r = run(BASELINE, noise, None, seed, base, "windowed@wholeclip", experiment="calibration")
            r["cells"].append({**r["cells"][0], "read": "windowed", "acc": base - 0.1})
            for c in r["cells"]:
                c["val_acc"] = c["acc"] + 0.01
            r["spec"]["protocol"] = "V"
            runs[f"V/{noise:g}db/s{seed}/baseline"] = r
            r = run(COUPLED, noise, 1.0, seed, net, "windowed", experiment="calibration")
            r["cells"][0]["val_acc"] = net + 0.01
            r["spec"]["protocol"] = "V"
            runs[f"V/{noise:g}db/g1/s{seed}/{rec.COUPLED_LABEL}"] = r
    (tmp_path / "calibration-recognition.json").write_text(json.dumps({"runs": runs}))
    rows = sm.calibration(rec.load())
    assert {(r["arm"], r["read"]) for r in rows} == {("baseline", "windowed@wholeclip"), (rec.COUPLED_LABEL, "windowed")}
    by = {(r["arm"], r["noise"]): r for r in rows}
    assert by[("baseline", -10.0)]["validation"]["mean"] == pytest.approx(96.0)
    assert [by[(rec.COUPLED_LABEL, n)]["in_window"] for n in (-10.0, 0.0, 10.0)] == [False, True, False]


def test_the_new_comparison_families_are_empty_on_a_record_without_their_experiments(recorded):
    cells = rec.load()
    assert sm.exact_drive(cells) == [] and sm.reservoir_baselines(cells) == [] and sm.projection_methods(cells) == []
    assert sm.selections(cells) == [] and sm.validation_accuracies(cells) == []
    assert sm.calibration(cells) == [] and sm.selection_profiles(cells) == [] and sm.clean(cells) == []
    assert sm.row_readout(cells) == [] and sm.band_readout(cells) == []
    assert sm.amplitude_views(cells) == [] and sm.stiffness(cells) == []

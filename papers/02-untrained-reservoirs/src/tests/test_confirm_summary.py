"""The raw-results summary on a small hand-built record whose answers are known."""

import json

import numpy as np
import pytest
import torch

from harness.confirm import readout as ro
from harness.confirm import score as sc
from harness.confirm import summary as sm

N_TEST = 400
FIELD = {"kind": "field", "physics": "kuramoto", "boundary": "torus", "omega": "random", "damping": 0.3,
         "clamp": 1.0, "channels": 4, "severed": False, "arch": ""}
FLOOR = {**FIELD, "kind": "floor"}


def bits(acc: float, seed: int) -> str:
    correct = np.zeros(N_TEST, dtype=bool)
    correct[np.random.default_rng(seed).choice(N_TEST, int(round(acc * N_TEST)), replace=False)] = True
    return ro.pack(torch.from_numpy(correct))


def run(arm, noise, gain, seed, acc, read, tier="tier1", fold=-1):
    spec = {"tier": tier, "task": "recognition", "drive": "envelope", "noise_db": noise, "gain": gain,
            "seed": seed, "arm": arm, "pair": [], "span": "fixed", "fold": fold}
    cells = [{"read": read, "width": 192, "n_train": 2048, "acc": acc, "correct": bits(acc, 7 * seed + fold + 2)}]
    return {"spec": spec, "cells": cells, "n_test": N_TEST}


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    runs = {}
    for noise in (0.0, 5.0):
        for seed in (0, 1, 2):
            runs[f"A/{noise:g}db/s{seed}/floor"] = run(FLOOR, noise, None, seed, 0.70, "windowed@wholeclip")
            for gain in (1.0, 2.0):
                acc = 0.78 + 0.01 * seed
                runs[f"A/{noise:g}db/g{gain:g}/s{seed}/{sc.FIELD_LABEL}"] = run(FIELD, noise, gain, seed, acc,
                                                                                  "windowed")
    root = tmp_path / "confirmatory"
    root.mkdir(parents=True)
    (root / "tier1-recognition-envelope.json").write_text(json.dumps({"runs": runs}))
    return root


def test_an_accuracy_is_its_mean_spread_and_every_replicate_in_points():
    s = sm.spread({"seed0": 0.78, "seed1": 0.79, "seed2": 0.80})
    assert s["mean"] == pytest.approx(79.0) and s["sd"] == pytest.approx(1.0) and s["n"] == 3
    assert s["values"] == pytest.approx({"seed0": 78.0, "seed1": 79.0, "seed2": 80.0})


def test_the_network_is_compared_with_the_gainless_baseline_at_every_gain(recorded):
    rows = [r for r in sm.tier1(sc.load())
            if r["comparison"] == "coupled oscillator network minus the spectrogram-only baseline, whole clip"]
    assert {(r["noise"], r["gain"]) for r in rows} == {(0.0, 1.0), (0.0, 2.0), (5.0, 1.0), (5.0, 2.0)}
    for r in rows:
        assert r["mean"] == pytest.approx(9.0) and r["sd"] == pytest.approx(1.0)
        assert r["values"] == pytest.approx({"seed0": 8.0, "seed1": 9.0, "seed2": 10.0})
        assert r["ci95"][0] < 9.0 < r["ci95"][1]


def test_folds_are_replicates_and_their_test_clips_are_pooled_not_averaged():
    a = [sc.Cell("becker", "recognition", "envelope", None, 1.0, 0, FIELD, "f", (), "windowed", 192, 18000,
                 acc, bits(acc, f), N_TEST, f) for f, acc in enumerate((0.80, 0.90))]
    b = [sc.Cell("becker", "recognition", "envelope", None, None, 0, FLOOR, "floor", (), "windowed", 192, 18000,
                 0.75, bits(0.75, 10 + f), N_TEST, f) for f in range(2)]
    r = sm.paired(list(zip(a, b, strict=True)))
    assert r["values"] == pytest.approx({"fold0": 5.0, "fold1": 15.0}) and r["mean"] == pytest.approx(10.0)
    assert r["ci95"][0] < 10.0 < r["ci95"][1]


def test_the_summary_runs_before_every_tier_is_in_and_the_record_ignores_it(recorded):
    sm.main([])
    text = (recorded / "summary.md").read_text()
    assert "(not run yet)" in text and "+9.00 ± 1.00" in text
    assert len(sc.load()) == 18                  # summary.json sits beside the record but is not read as one

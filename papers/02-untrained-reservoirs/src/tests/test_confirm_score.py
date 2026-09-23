"""The confirmatory scorer on a small hand-built record whose answers are known."""

import json

import numpy as np
import pytest

from harness.confirm import readout as ro
from harness.confirm import run as rn
from harness.confirm import score as sc

N_TEST = 400


def bits(acc: float, seed: int) -> str:
    rng = np.random.default_rng(seed)
    correct = np.zeros(N_TEST, dtype=bool)
    correct[rng.choice(N_TEST, int(round(acc * N_TEST)), replace=False)] = True
    import torch
    return ro.pack(torch.from_numpy(correct))


def record(label_arm: dict, noise, gain, seed, acc, read="windowed", task="recognition", pair=()):
    spec = {"tier": "tier1", "task": task, "drive": "envelope", "noise_db": noise, "gain": gain, "seed": seed,
            "arm": label_arm, "pair": list(pair), "span": "fixed"}
    cells = [{"read": read, "width": 192, "n_train": 2048, "acc": acc, "correct": bits(acc, seed * 7 + int(acc * 1000))}]
    return spec, {"spec": spec, "cells": cells, "n_test": N_TEST}


FIELD = {"kind": "field", "physics": "kuramoto", "boundary": "torus", "omega": "random", "damping": 0.3,
         "clamp": 1.0, "channels": 4, "severed": False, "arch": ""}
FLOOR = {**FIELD, "kind": "floor"}


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    runs = {}
    for noise in (0.0, 5.0):
        for seed in (0, 1, 2):
            runs[f"A/{noise:g}db/s{seed}/floor"] = record(FLOOR, noise, None, seed, 0.70, read="windowed@wholeclip")[1]
            for gain in (1.0, 2.0):
                runs[f"A/{noise:g}db/g{gain:g}/s{seed}/{sc.FIELD_LABEL}"] = record(FIELD, noise, gain, seed, 0.80)[1]
    path = tmp_path / "confirmatory" / "tier1-recognition-envelope.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"group": "tier1-recognition-envelope", "runs": runs}))
    return tmp_path


def test_a_ten_point_gain_at_every_condition_supports_h1(recorded):
    out = sc.hypotheses_h1_to_h3(sc.load())["H1"]
    assert out["verdict"] == "supported"
    for r in out["conditions"].values():
        assert r["diff"] == pytest.approx(10.0) and r["ci"][0] < 10.0 < r["ci"][1]
        assert set(r["seeds"]) == {0, 1, 2}


def test_a_bar_needs_every_seed_on_its_side():
    assert sc.meets({"n_pairs": 3, "diff": 4.0, "seeds": {0: 5.0, 1: 5.0, 2: 2.0}}, 3.0)
    assert not sc.meets({"n_pairs": 3, "diff": 4.0, "seeds": {0: 7.0, 1: 7.0, 2: -2.0}}, 3.0)
    assert not sc.meets({"n_pairs": 2, "diff": 9.0, "seeds": {0: 9.0, 1: 9.0}}, 3.0)
    assert sc.meets({"n_pairs": 3, "diff": -4.0, "seeds": {0: -4.0, 1: -3.5, 2: -4.5}}, -3.0)


def test_verdicts_name_all_three_outcomes():
    assert sc.verdict({"a": True, "b": True}) == "supported"
    assert sc.verdict({"a": False, "b": False}) == "refuted"
    assert sc.verdict({"a": True, "b": False}) == "mixed"
    assert sc.verdict({}) == "not run"


def test_an_arm_that_was_not_run_is_not_run_rather_than_refuted(recorded):
    assert sc.hypotheses_h1_to_h3(sc.load())["H3"]["verdict"] == "not run"

"""The summary on a small hand-built record whose answers are known, and paper 02's cells folded in."""

import json

import numpy as np
import pytest
import torch

from harness.confirm import plan
from harness.confirm import readout as ro
from harness.confirm import summary as sm
from harness.confirm.arms import Arm

N_TEST = 400
FIELD_LABEL = "field-kuramoto-torus-random-lam0.3-clamp1"


def bits(acc: float, seed: int) -> str:
    correct = np.zeros(N_TEST, dtype=bool)
    correct[np.random.default_rng(seed).choice(N_TEST, int(round(acc * N_TEST)), replace=False)] = True
    return ro.pack(torch.from_numpy(correct))


def run(arm: Arm, noise, gain, seed, acc, read, tier="size"):
    spec = {"tier": tier, "task": "recognition", "drive": "envelope", "noise_db": noise, "gain": gain,
            "seed": seed, "arm": arm.as_dict(), "span": "fixed"}
    # recorded as paper 02 records: no projection tag, so the summary reads them as the fixed projection
    cells = [{"read": read, "width": 192, "effective_width": 192, "n_train": 2048, "acc": acc,
              "correct": bits(acc, 7 * seed + 2)}]
    return {"spec": spec, "cells": cells, "n_test": N_TEST, "native_widths": {read: 10**6}}


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(plan, "PAPER02_RECORD", tmp_path / "paper02")
    runs, old = {}, {}
    for noise in (0.0, 5.0):
        for seed in (0, 1, 2):
            runs[f"A/{noise:g}db/s{seed}/floor-32x32"] = run(Arm("floor", grid=32), noise, None, seed, 0.70,
                                                              "windowed@wholeclip")
            for gain in (1.0, 2.0):
                net = Arm("field", channels=8, grid=32)
                runs[f"A/{noise:g}db/g{gain:g}/s{seed}/{net.label()}"] = run(net, noise, gain, seed,
                                                                             0.78 + 0.01 * seed, "windowed")
                # paper 02's 16 x 16, 4-channel network, recorded in its own tier 1 file
                old[f"A/{noise:g}db/g{gain:g}/s{seed}/{FIELD_LABEL}"] = run(Arm("field"), noise, gain, seed, 0.75,
                                                                           "windowed", tier="tier1")
    (tmp_path / "confirmatory").mkdir(parents=True)
    (tmp_path / "confirmatory" / "size-envelope-32x32.json").write_text(json.dumps({"runs": runs}))
    (tmp_path / "paper02").mkdir()
    (tmp_path / "paper02" / "tier1-recognition-envelope.json").write_text(json.dumps({"runs": old}))
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


def test_paper_02s_cells_are_read_from_its_record_under_paper_03s_tier(recorded):
    acc = sm.summary()["accuracy"]
    reused = [r for r in acc if r["arm"] == FIELD_LABEL]
    assert reused and all(r["tier"] == "size" and r["source"] == ["paper 02"] for r in reused)
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

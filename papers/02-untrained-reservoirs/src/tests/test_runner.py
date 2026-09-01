"""End-to-end contracts for `harness.runner`.

Quick smoke passes that exercise the whole path — data, field, training,
evaluation, serialization — for each task and frontend, so a break in the
wiring surfaces here rather than eight hours into a matrix."""

import math
import sys

import torch

sys.path.insert(0, ".")
from harness.runner import main as run_main

TWO_PI = 2 * math.pi


def only_run(root) -> dict:
    """The single run a smoke pass recorded, wherever the store filed it."""
    import json
    groups = [p for p in root.rglob("*.json")]
    assert len(groups) == 1, groups
    runs = json.loads(groups[0].read_text())["runs"]
    assert len(runs) == 1, list(runs)
    return next(iter(runs.values()))


def test_quick_end_to_end_records_a_run(tmp_path, monkeypatch):
    """The whole path in one pass: data, field, training, evaluation, and the
    record — so a break in the wiring surfaces here rather than eight hours
    into a sweep. Writes into a scratch record via the documented override."""
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    run_main(["--quick", "--task", "tones", "--arms", "kuramoto,frozen,designed",
              "--epochs", "1", "--seeds", "0", "--threads", "2"])
    rows = only_run(tmp_path)["rows"]
    assert {r["arm"] for r in rows} == {"kuramoto", "frozen", "designed"}
    assert all(0.0 <= r["ridge_acc"] <= 1.0 for r in rows)
    # settle column and parity column ride every eval row
    assert all(0.0 <= r["ridge_acc_settle"] <= 1.0 for r in rows)
    assert all(0.0 <= r["ridge_acc_parity"] <= 1.0 for r in rows)


def test_quick_end_to_end_on_an_open_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    run_main(["--quick", "--task", "am", "--arms", "kuramoto,frozen",
              "--boundary", "cylinder", "--epochs", "1", "--seeds", "0", "--threads", "2"])
    rows = only_run(tmp_path)["rows"]
    assert {r["arm"] for r in rows} == {"kuramoto", "frozen"}


def test_frontend_quad_cli_unlocked():
    from harness.runner import parse_args
    args = parse_args(["--task", "digits", "--frontend", "quad"])
    assert args.frontend == "quad" and args.frames == 61


def test_frontend_carrier_digits_contract():
    # Matrix D: carrier bandpass rows at sample rate, tvalid in samples
    import pytest

    from harness.runner import make_data, parse_args
    from harness.stimuli.digits import DIGIT_BANK_PATH
    if not DIGIT_BANK_PATH.exists():
        pytest.skip("digit bank not built")
    args = parse_args(["--task", "digits", "--frontend", "carrier"])
    rows, _, labels, k, tvalid = make_data(args, 8, 1234)
    assert rows.shape == (8, 16000, args.grid) and k == 10
    assert torch.isfinite(rows).all()
    # tvalid is TRUE SAMPLE length (>1600), not hop frames (<= 62)
    assert (tvalid > 1600).all() and (tvalid <= 16000).all()
    # band-split property: rows are not all identical across bands
    assert not torch.allclose(rows[:, :, 0], rows[:, :, 8], atol=1e-6)
    # same clip draw as the mag frontend at the same seed (waveform-level parity)
    args_mag = parse_args(["--task", "digits", "--frontend", "mag"])
    _, _, labels_mag, _, _ = make_data(args_mag, 8, 1234)
    assert torch.equal(labels, labels_mag)


def test_omega_uniform_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    from harness.runner import parse_args, run_matrix
    args = parse_args(["--task", "tones", "--quick", "--arms", "frozen",
                       "--omega-uniform"])
    rows = run_matrix(args)
    assert rows and abs(rows[0]["omega_std"]) < 1e-7  # exactly uniform


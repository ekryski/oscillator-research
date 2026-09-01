"""Contracts for `harness.baselines`.

The conventional references only mean something at matched budget, so their
parameter counts are asserted against the physics budget, and they are held
to the identical feature/ridge/settle contract as every other arm."""

import math
import sys

import torch

sys.path.insert(0, ".")
from harness.models import (
    GRUBaseline,
)
from harness.models.baselines import CNNBaseline, S4DBaseline, TransformerBaseline
from harness.runner import main as run_main

TWO_PI = 2 * math.pi


def test_conventional_references_windowed_parity():
    # item-7 parity: GRU/TCN report windowed features under the same contract
    # as OscillatorField (windows x feat_dim), so windowed-primary rounds stay arm-fair
    from harness import TCNBaseline
    torch.manual_seed(0)
    rows = torch.randn(3, 64, 16)
    for cls in (GRUBaseline, TCNBaseline):
        m = cls(grid=16, n_classes=5)
        f, fw = m.features(rows), m.features_windowed(rows, windows=4)
        assert f.shape[1] == m.feat_dim and fw.shape[1] == 4 * m.feat_dim
        assert torch.isfinite(fw).all()


def test_baseline_minis_budget_and_contract():
    # exp-1 minis: within 15% of the 2,048 physics budget; full arm
    # contract (features/windowed/settle/tvalid/forward, frozen probe); causal.
    torch.manual_seed(1)
    rows = torch.rand(3, 64, 16) * 0.5
    tv = torch.tensor([30, 45, 64])
    full = torch.full((3,), 64, dtype=torch.long)
    for cls in (CNNBaseline, TransformerBaseline, S4DBaseline):
        torch.manual_seed(0)
        m = cls(grid=16, n_classes=5)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert abs(n - 2048) / 2048 < 0.15, (cls.__name__, n)
        f = m.features(rows)
        assert f.shape == (3, m.feat_dim) and torch.isfinite(f).all()
        # tvalid == T matches unmasked up to the masked-std eps (sqrt(1e-8)
        # = 1e-4 for a dead unit's exactly-zero variance — the NaN guard)
        assert torch.allclose(m.features(rows, tvalid=full), f, atol=2e-4)
        assert not torch.allclose(m.features(rows, tvalid=tv)[0], f[0], atol=1e-3)
        fw = m.features_windowed(rows, windows=4, tvalid=tv)
        assert fw.shape == (3, 4 * m.feat_dim) and torch.isfinite(fw).all()
        fs = m.features_settle(rows)
        assert fs.shape == (3, m.feat_dim) and torch.isfinite(fs).all()
        assert not torch.allclose(fs, f, atol=1e-3)
        out = m(rows, tv)
        assert out.shape == (3, 5) and torch.isfinite(out).all()
        assert not m.probe_w.requires_grad  # frozen probe is a buffer
        # causal: perturbing the LAST frame must not change earlier outputs
        r2 = rows.clone()
        r2[:, -1] += 1.0
        h1, h2 = m._hidden(rows), m._hidden(r2)
        assert torch.allclose(h1[:, :-1], h2[:, :-1], atol=1e-5), cls.__name__
        assert not torch.allclose(h1[:, -1], h2[:, -1], atol=1e-4), cls.__name__


def test_baseline_minis_quick_end_to_end_train(tmp_path, monkeypatch):
    # the three minis register as arms, train (loss drops), and get the full
    # eval column set under the identical protocol every other arm gets
    import json
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    run_main(["--quick", "--task", "tones", "--arms", "cnn,transformer,s4d",
              "--epochs", "3", "--seeds", "0", "--threads", "2"])
    group = next(p for p in tmp_path.rglob("*.json"))
    rows = next(iter(json.loads(group.read_text())["runs"].values()))["rows"]
    assert {r["arm"] for r in rows} == {"cnn", "transformer", "s4d"}
    for r in rows:
        assert r["loss_drop"] > 0.0, r["arm"]  # measured: +6%/+56%/+11%
        assert 0.0 <= r["ridge_acc_settle"] <= 1.0
        assert 0.0 <= r["ridge_acc_parity"] <= 1.0
    curves = list((tmp_path / "training-curves").glob("*.csv"))
    assert len(curves) == 3, curves

"""The readout-sufficiency ladder.

Ridge vs MLP vs tiny-transformer heads on IDENTICAL windowed features from one
scored-condition frozen run and its floor twin, swept over training-set size.
Its job is to test the weakest-reader design choice rather than assume it: if a
stronger head extracts substantially more from the same features, the linear
probe was leaving field structure on the table.

Sizes above the bank's 9,600 unique training clips are an augmentation regime —
sampling is with replacement and repeats get fresh noise draws — and are
labelled as such in the output.

    uv run python scripts/readout_ladder.py                    # the full ladder
    uv run python scripts/readout_ladder.py --sizes 2048 4096  # a subset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from harness.measurement.floor import floor_features_windowed
from harness.measurement.probe import fit_ridge_probe
from harness.models import OscillatorField
from harness.runner import TEST_SEED, make_data, parse_args
from harness.utils.paths import RESULTS_DIR

OUT = RESULTS_DIR / "baselines" / "readout-ladder.json"
DEFAULT_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
UNIQUE_TRAIN_CLIPS = 9_600  # above this, sampling with replacement augments
N_VAL, N_TEST = 256, 512
EPOCHS, BATCH, LR, PATIENCE = 200, 64, 1e-3, 20
WINDOWS = 4

# the scored run the ladder interrogates: the primary condition's crown config
FIELD = dict(coupling="kuramoto", boundary="torus", damping=0.3, spectral_clamp=1.0,
            gain=2.0, channels=4, grid=16, n_classes=10, probe_seed=5000, seed=0)
FIELD_ARGV = ["--task", "digits", "--frontend", "mag", "--noise-db", "0", "--gain", "2.0",
             "--damping", "0.3", "--clamp", "1.0", "--coupling", "kuramoto",
             "--boundary", "torus"]


def featurize(args, n: int, gen_seed: int, model: OscillatorField):
    """(field features, floor features, labels) for n clips from one pool."""
    rows, _, y, _, tv = make_data(args, n, gen_seed)
    with torch.no_grad():
        feats = torch.cat([model.features_windowed(rows[i:i + 64], WINDOWS,
                                                   tvalid=tv[i:i + 64])
                           for i in range(0, n, 64)])
    return feats, floor_features_windowed(rows, WINDOWS), y


def train_head(head, f_tr, y_tr, f_va, y_va, f_te, y_te, seed=0) -> dict:
    """Early-stopped CE training on standardized features; reports the honest
    train-test gap alongside the test accuracy, since every head here carries
    more parameters than the entire oscillator field."""
    torch.manual_seed(seed)
    mu, sd = f_tr.mean(0, keepdim=True), f_tr.std(0, keepdim=True).clamp_min(1e-6)
    f_tr, f_va, f_te = (f_tr - mu) / sd, (f_va - mu) / sd, (f_te - mu) / sd
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    best_va, best_state, stale, ep = -1.0, None, 0, 0
    for ep in range(EPOCHS):
        head.train()
        perm = torch.randperm(len(f_tr))
        for i in range(0, len(f_tr), BATCH):
            idx = perm[i:i + BATCH]
            loss = nn.functional.cross_entropy(head(f_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            va = (head(f_va).argmax(1) == y_va).float().mean().item()
        if va > best_va:
            best_va, stale = va, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        tr = (head(f_tr).argmax(1) == y_tr).float().mean().item()
        te = (head(f_te).argmax(1) == y_te).float().mean().item()
    return dict(test=te, train=tr, gap=tr - te, val=best_va,
                params=sum(p.numel() for p in head.parameters()), epochs=ep + 1)


class WindowTransformer(nn.Module):
    """4 window-tokens -> 2-layer 2-head d=64 encoder -> mean -> linear."""

    def __init__(self, feat_dim: int, n_classes: int = 10, d: int = 64):
        super().__init__()
        assert feat_dim % WINDOWS == 0
        self.tok_dim = feat_dim // WINDOWS
        self.proj = nn.Linear(self.tok_dim, d)
        layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=2 * d,
                                           batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Linear(d, n_classes)

    def forward(self, x):
        b = x.shape[0]
        t = self.proj(x.view(b, WINDOWS, self.tok_dim))
        return self.out(self.enc(t).mean(dim=1))


def readers(feat_dim: int) -> dict:
    return {
        "mlp": lambda: nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, 10)),
        "transformer": lambda: WindowTransformer(feat_dim),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="readout-sufficiency ladder")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                    help="training-set sizes to sweep")
    ap.add_argument("--head-seeds", type=int, default=3,
                    help="head-init replicates per (size, reader)")
    a = ap.parse_args(argv)

    args = parse_args(FIELD_ARGV)
    torch.manual_seed(0)
    model = OscillatorField(**FIELD)
    biggest = max(a.sizes)
    f_tr, fl_tr, y_tr = featurize(args, biggest + N_VAL, 1000, model)
    f_te, fl_te, y_te = featurize(args, N_TEST, TEST_SEED, model)

    out: dict[str, dict] = {"run": {}, "floor": {}}
    for tag, tr_x, te_x in (("run", f_tr, f_te), ("floor", fl_tr, fl_te)):
        va_x, va_y = tr_x[biggest:], y_tr[biggest:]
        for size in sorted(a.sizes):
            x, y = tr_x[:size], y_tr[:size]
            r = fit_ridge_probe(x, y, te_x, y_te, 10)
            entry: dict = {"ridge": {"test": r["acc"], "lam": r["lam"]},
                           "augmented": size > UNIQUE_TRAIN_CLIPS}
            for name, mk in readers(tr_x.shape[1]).items():
                runs = [train_head(mk(), x, y, va_x, va_y, te_x, y_te, seed=s)
                        for s in range(a.head_seeds)]
                tests = [r["test"] for r in runs]
                entry[name] = {"tests": tests, "mean": sum(tests) / len(tests),
                               "spread": max(tests) - min(tests),
                               "gap": sum(r["gap"] for r in runs) / len(runs),
                               "params": runs[0]["params"]}
            out[tag][str(size)] = entry
            best = {k: round(v["mean"] if "mean" in v else v["test"], 4)
                    for k, v in entry.items() if isinstance(v, dict)}
            print(f"{tag} n={size}: {json.dumps(best)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

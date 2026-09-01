"""Conventional references under the TRAINED-HEAD protocol.

The field arms train against a frozen random readout direction, because that is
the physics-symmetric objective: gradients have to reshape the dynamics, not the
reader. Conventional networks are normally trained the other way, with a learned
classifier head, and the difference is not cosmetic — the frozen-probe objective
turns out to be hostile to ReLU feed-forward architectures, which collapse under
it while training fine conventionally at the same size.

Reporting only one protocol would therefore either handicap the baselines or
abandon the symmetry the field arms need. So both are run, and both are
reported. This script is the trained-head half; `harness.runner --arms <arch>`
is the frozen-probe half.

Everything downstream of the trained head is identical to every other arm: the
same features, the same closed-form ridge, the same windowed metric, the same
seeds.

    uv run python scripts/trained_head_baselines.py --noise-db 0
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

from harness.measurement.probe import fit_ridge_probe
from harness.models import GRUBaseline, TCNBaseline
from harness.models.baselines import CNNBaseline, S4DBaseline, TransformerBaseline
from harness.runner import TEST_SEED, make_data, parse_args
from harness.utils.paths import RESULTS_DIR

ARCHS = {"cnn": CNNBaseline, "tcn": TCNBaseline, "s4d": S4DBaseline,
         "gru": GRUBaseline, "transformer": TransformerBaseline}
SEEDS = (0, 1, 2)
EPOCHS, BATCH, LR, WINDOWS = 30, 64, 3e-3, 4
N_CLASSES = 10


class TrainedHead(nn.Module):
    """A baseline plus a learned linear classifier — conventional practice."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.feat_dim, N_CLASSES)

    def forward(self, rows: torch.Tensor, tvalid: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.backbone.features(rows, tvalid))


def train(model: nn.Module, rows, y, tvalid, seed: int) -> float:
    """Train to convergence; return the relative train-loss drop.

    The drop is the optimization-health gate: an arm that never learned is an
    optimization failure, which is a different claim from "this architecture
    cannot do the task", and the two are not allowed to be confused.
    """
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS * ((len(rows) + BATCH - 1) // BATCH), eta_min=0.1 * LR)
    gen = torch.Generator().manual_seed(seed)
    first = last = None
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(rows), generator=gen)
        total = 0.0
        for i in range(0, len(rows), BATCH):
            idx = perm[i:i + BATCH]
            loss = nn.functional.cross_entropy(model(rows[idx], tvalid[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        last = total / len(rows)
        first = first if first is not None else last
    model.eval()
    return (first - last) / first if first else 0.0


@torch.no_grad()
def features(backbone: nn.Module, rows, tvalid, windows: int | None = None):
    out = []
    for i in range(0, len(rows), BATCH):
        chunk, tv = rows[i:i + BATCH], tvalid[i:i + BATCH]
        out.append(backbone.features_windowed(chunk, windows, tv) if windows
                   else backbone.features(chunk, tv))
    return torch.cat(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="trained-head conventional references")
    ap.add_argument("--noise-db", type=float, required=True)
    ap.add_argument("--gain", type=float, default=2.0)
    a = ap.parse_args(argv)

    args = parse_args(["--task", "digits", "--frontend", "mag",
                       "--noise-db", str(a.noise_db), "--gain", str(a.gain)])
    rows_tr, _, y_tr, _, tv_tr = make_data(args, args.n_train, 0)
    rows_te, _, y_te, _, tv_te = make_data(args, args.n_test, TEST_SEED)

    out: dict[str, list[dict]] = {}
    for name, cls in ARCHS.items():
        out[name] = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            backbone = cls(grid=args.grid, n_classes=N_CLASSES, probe_seed=5000 + seed)
            model = TrainedHead(backbone)
            drop = train(model, rows_tr, y_tr, tv_tr, seed)
            std = fit_ridge_probe(features(backbone, rows_tr, tv_tr), y_tr,
                                  features(backbone, rows_te, tv_te), y_te, N_CLASSES)
            win = fit_ridge_probe(features(backbone, rows_tr, tv_tr, WINDOWS), y_tr,
                                  features(backbone, rows_te, tv_te, WINDOWS), y_te, N_CLASSES)
            out[name].append({"seed": seed, "ce_drop": drop,
                              "ridge": std["acc"], "windowed": win["acc"]})
            print(f"{name:12s} seed {seed}  loss drop {drop:5.1%}  "
                  f"ridge {std['acc']:.3f}  windowed {win['acc']:.3f}")

    dst = RESULTS_DIR / "baselines" / f"conventional-trainedhead-{a.noise_db:g}db.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()

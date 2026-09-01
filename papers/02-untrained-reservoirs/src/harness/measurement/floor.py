"""The no-dynamics floor: the identical ridge on the frontend features.

The single most clarifying baseline, and the reason every accuracy in the paper
decomposes into representation + dynamics: whatever an arm scores above this
line is what its dynamics bought, and an arm below it is destroying
information. No training, no field, no parameters — per-band time-mean,
time-std, and mean |delta| (48 features at G=16) through the same deterministic
ridge protocol every arm gets.

    uv run python -m harness.measurement.floor --task digits --noise-db 0 --windows 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.measurement.probe import fit_ridge_probe
from harness.results import results_root


def floor_features(rows: torch.Tensor) -> torch.Tensor:
    """[B,T,G] -> [B, 3G]: per-band mean, std, mean |frame delta|."""
    d = (rows[:, 1:] - rows[:, :-1]).abs().mean(dim=1)
    return torch.cat((rows.mean(dim=1), rows.std(dim=1), d), dim=1)


def floor_features_windowed(rows: torch.Tensor, windows: int) -> torch.Tensor:
    """Windowed-parity floor: per-window floor stats, concatenated — the
    statics baseline any windowed-primary round must be scored against."""
    t = rows.shape[1]
    assert t // windows >= 2, f"window too short: {t} frames / {windows} windows"
    return torch.cat([floor_features(rows[:, i * t // windows:(i + 1) * t // windows])
                      for i in range(windows)], dim=1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="the no-dynamics raw-feature floor")
    ap.add_argument("--task", default="digits")
    ap.add_argument("--noise-db", type=float, default=0.0)
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--n-test", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", type=int, default=0,
                    help="ALSO report the windowed-parity floor (0 = off)")
    ap.add_argument("--frontend", default="mag",
                    help="digits frontend (mag | quad | carrier); every cross-frontend "
                         "comparison needs its floor ON its own representation")
    ap.add_argument("--pair", default="3,7",
                    help="digitpairs: the two digits for the order-task floor")
    ap.add_argument("--record", action="store_true",
                    help="merge this floor into results/baselines/floors.json, where "
                         "the scoring module reads it from")
    a = ap.parse_args(argv)
    # imported here, not at module scope: the floor's feature functions are
    # pure, and the runner pulls in the whole model stack
    from harness.runner import TEST_SEED, make_data
    from harness.runner import parse_args as runner_args
    args = runner_args(["--task", a.task, "--noise-db", str(a.noise_db),
                        "--n-train", str(a.n_train), "--n-test", str(a.n_test),
                        "--frontend", a.frontend, "--pair", a.pair])
    rows_tr, _, y_tr, k, tv_tr = make_data(args, args.n_train, a.seed)
    rows_te, _, y_te, _, tv_te = make_data(args, args.n_test, TEST_SEED)
    r = fit_ridge_probe(floor_features(rows_tr), y_tr, floor_features(rows_te), y_te, k)
    print(f"FLOOR task={a.task} noise={a.noise_db:+.0f} frames={args.frames} seed={a.seed}: "
          f"acc {r['acc']:.3f} (chance {1 / k:.3f}, lam {r['lam']})")
    rw = None
    if a.windows:
        rw = fit_ridge_probe(floor_features_windowed(rows_tr, a.windows), y_tr,
                             floor_features_windowed(rows_te, a.windows), y_te, k)
        print(f"FLOOR-WINDOWED (x{a.windows}): acc {rw['acc']:.3f} (lam {rw['lam']})")
    if a.record:
        record_floor(a, r, rw)


DRIVE = {"mag": "envelope", "quad": "quadrature", "carrier": "carrier"}


def record_floor(a, standard: dict, windowed: dict | None) -> None:
    """Merge one floor into results/baselines/floors.json.

    Floors need the stimulus bank, so they are measured once here rather than
    recomputed by every scoring pass. The key names the representation and the
    noise level, because a floor is only ever comparable to runs that share
    both: the carrier path's band window discards content the mel path keeps,
    so its floor is genuinely a different number, not a worse one.
    """
    out = results_root() / "baselines" / "floors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    floors = json.loads(out.read_text()) if out.exists() else {}
    key = f"{DRIVE[a.frontend]}-{a.noise_db:g}db"
    if a.task == "digitpairs":
        key = f"{key}-order-pair{a.pair.replace(',', '')}"
    entry = {"standard": standard["acc"], "chance": 1.0 / (2 if a.task == "digitpairs" else 10),
             "n_train": a.n_train, "n_test": a.n_test, "seed": a.seed}
    if windowed:
        entry["windowed"] = windowed["acc"]
    floors[key] = entry
    out.write_text(json.dumps(dict(sorted(floors.items())), indent=1) + "\n")
    print(f"recorded floor '{key}' in {out}")


if __name__ == "__main__":
    main()

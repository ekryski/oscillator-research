"""The run record, read back: every recorded cell with what it belongs to.

The summary and the figures read the record through this module. Nothing here
decides whether a result passes: it loads cells, unpacks their per-clip
correctness, and reports where the order task's spectrogram-only baseline sits
against chance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from harness.experiment import run as rn
from harness.experiment.arms import Arm
from harness.experiment.readout import unpack

PRIMARY_WIDTH, PRIMARY_SIZE = 192, 2048
PRIMARY_READ = {"recognition": "windowed", "order": "pooled"}
SEEDS = (0, 1, 2)
BOOTSTRAP, CONFIDENCE, BOOT_SEED = 2000, 0.95, 20260923
COUPLED_LABEL = "coupled-kuramoto-torus-random-restoring0.3-ceiling1"


@dataclass(frozen=True)
class Cell:
    """One recorded accuracy, with what it belongs to."""
    tier: str
    task: str
    pathway: str
    noise: float | None
    gain: float | None
    seed: int
    arm: dict
    label: str
    pair: tuple
    read: str
    width: object
    n_train: int
    acc: float
    bits: str | None
    n_test: int
    fold: int = -1
    projection: str = "fixed"


def load(groups: list[str] | None = None) -> list[Cell]:
    root = rn.record_root()
    paths = sorted(root.glob("*.json")) if groups is None else [root / f"{g}.json" for g in groups]
    out = []
    for path in paths:
        if path.name in ("gates.json", "summary.json") or not path.exists():
            continue
        for rec in json.loads(path.read_text())["runs"].values():
            s = rec["spec"]
            label = Arm(**s["arm"]).label()
            if s.get("span", "fixed") != "fixed":
                label += "@clipspan"
            for c in rec["cells"]:
                out.append(Cell(s["tier"], s["task"], s["pathway"], s["noise_db"], s["gain"], s["seed"],
                                s["arm"], label, tuple(s["pair"]), c["read"], c["width"], c["n_train"],
                                c["acc"], c.get("correct"), rec["n_test"], s.get("fold", -1),
                                c.get("projection", "fixed")))
    return out


def bits(c: Cell) -> np.ndarray:
    if c.bits is None:
        raise ValueError(f"{c.label} {c.read} {c.width} has no per-clip record")
    return unpack(c.bits, c.n_test).numpy().astype(np.float64)


def baseline_at_chance(cells: list[Cell]) -> dict:
    """The order task's spectrogram-only baseline, per pair and noise level: its accuracy, the 95% interval
    from resampling test clips, and whether that interval contains chance (50%)."""
    base = [c for c in cells if c.tier == "tier1" and c.task == "order" and c.arm["kind"] == "baseline"
            and c.width == PRIMARY_WIDTH and c.n_train == PRIMARY_SIZE]
    out = {}
    for pair in sorted({c.pair for c in base}):
        for noise in sorted({c.noise for c in base}, key=lambda x: -1 if x is None else x):
            entry = {}
            for read in ("pooled@wholeclip", "pooled"):
                mine = [c for c in base if c.pair == pair and c.noise == noise and c.read == read]
                if len(mine) < len(SEEDS):
                    continue
                per_clip = np.mean([bits(c) for c in mine], axis=0)
                rng = np.random.default_rng(BOOT_SEED)
                boots = per_clip[rng.integers(0, len(per_clip), (BOOTSTRAP, len(per_clip)))].mean(axis=1)
                lo, hi = np.quantile(boots, [(1 - CONFIDENCE) / 2, (1 + CONFIDENCE) / 2])
                entry[read] = {"acc": float(per_clip.mean()), "ci": (float(lo), float(hi)),
                               "contains_chance": bool(lo <= 0.5 <= hi)}
            out[f"pair{pair[0]}{pair[1]}/{'clean' if noise is None else f'{noise:g}db'}"] = entry
    return out

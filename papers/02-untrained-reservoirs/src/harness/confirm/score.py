"""Score the confirmatory record against the registered bars (REGISTRATION.md, section 9).

    uv run python -m harness.confirm.score            # print every verdict, write verdicts.json

Written and committed before any registered run, so the analysis could not be
shaped by the numbers it reads. Every comparison is paired over the same test
clips: for two arms A and B, each clip gets d = mean over seeds of (A right -
B right), the reported difference is the mean of d, and its 95% interval comes
from resampling clips. A bar is met only if the three-seed mean clears it and
every seed's own difference has the same sign. A hypothesis is supported if its
bar is met at every discriminating condition, refuted if it is missed at every
one, and mixed otherwise.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from harness.confirm import run as rn
from harness.confirm.readout import unpack

PRIMARY_WIDTH, PRIMARY_SIZE = 192, 2048
PRIMARY_READ = {"recognition": "windowed", "order": "pooled"}
DISCRIMINATING_NOISE = (0.0, 5.0)
GAINS = (1.0, 2.0)
SEEDS = (0, 1, 2)
BOOTSTRAP, CONFIDENCE, BOOT_SEED = 2000, 0.95, 20260923
BARS = {"H1": 3.0, "H2": 3.0, "H3": 3.0, "H5": 5.0, "H6_diff": 3.0, "H6_acc": 0.60, "H6_pairs": 3,
        "shape": 3.0, "omega": 5.0, "family": 3.0, "drive": 5.0}
REFERENCE = {"boundary": "torus", "omega": "random", "physics": "kuramoto"}


@dataclass(frozen=True)
class Cell:
    """One recorded accuracy, with what it belongs to."""
    tier: str
    task: str
    drive: str
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
        if path.name in ("gates.json", "verdicts.json", "summary.json") or not path.exists():
            continue
        for run_id, rec in json.loads(path.read_text())["runs"].items():
            s = rec["spec"]
            label = run_id.split("/")[-1] if s["arm"]["kind"] != "ann" else run_id.split("/")[-2]
            if s.get("span", "fixed") != "fixed":
                label += "@clipspan"
            for c in rec["cells"]:
                out.append(Cell(s["tier"], s["task"], s["drive"], s["noise_db"], s["gain"], s["seed"],
                                s["arm"], label, tuple(s["pair"]), c["read"], c["width"], c["n_train"],
                                c["acc"], c.get("correct"), rec["n_test"], s.get("fold", -1),
                                c.get("projection", "fixed")))
    return out


def primary(cells: list[Cell], task: str = "recognition", width=PRIMARY_WIDTH, n=PRIMARY_SIZE,
            floor_span: str = "@wholeclip") -> list[Cell]:
    """The primary cells: the whole-clip floor, and every other arm's own window."""
    base = PRIMARY_READ[task]
    want = lambda c: c.read == (base + floor_span if c.arm["kind"] == "floor" else base)  # noqa: E731
    return [c for c in cells if c.task == task and c.width == width and c.n_train == n and want(c)]


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------

def _bits(c: Cell) -> np.ndarray:
    if c.bits is None:
        raise ValueError(f"{c.label} {c.read} {c.width} has no per-clip record")
    return unpack(c.bits, c.n_test).numpy().astype(np.float64)


def compare(pairs: list[tuple[Cell, Cell]]) -> dict:
    """A minus B over matched cells: the mean difference in points, per-seed signs, a clip bootstrap.

    `pairs` may hold several matched cells per seed (twins, conditions); each
    seed's difference is the mean over its own pairs.
    """
    if not pairs:
        return {"n_pairs": 0}
    per_clip = np.mean([_bits(a) - _bits(b) for a, b in pairs], axis=0)
    by_seed = defaultdict(list)
    for a, b in pairs:
        by_seed[a.seed].append(a.acc - b.acc)
    seeds = {s: 100 * statistics.fmean(v) for s, v in sorted(by_seed.items())}
    rng = np.random.default_rng(BOOT_SEED)
    boots = per_clip[rng.integers(0, len(per_clip), (BOOTSTRAP, len(per_clip)))].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - CONFIDENCE) / 2, (1 + CONFIDENCE) / 2])
    return {"diff": 100 * float(per_clip.mean()), "ci": (100 * float(lo), 100 * float(hi)),
            "seeds": seeds, "n_pairs": len(pairs)}


def meets(result: dict, bar: float) -> bool:
    """The three-seed mean clears the bar, and every seed agrees in sign with it."""
    if not result.get("n_pairs") or len(result["seeds"]) < len(SEEDS):
        return False
    signs = {np.sign(v) for v in result["seeds"].values()}
    if bar >= 0:
        return result["diff"] >= bar and signs == {1.0}
    return result["diff"] <= bar and signs == {-1.0}


def verdict(outcomes: dict[str, bool]) -> str:
    if not outcomes:
        return "not run"
    if all(outcomes.values()):
        return "supported"
    return "refuted" if not any(outcomes.values()) else "mixed"


def _index(cells: list[Cell], key) -> dict:
    out = {}
    for c in cells:
        out[key(c)] = c
    return out


def conditions(gain_applies: bool = True) -> list[tuple]:
    return [(n, g) for n in DISCRIMINATING_NOISE for g in (GAINS if gain_applies else (None,))]


# ---------------------------------------------------------------------------
# The hypotheses
# ---------------------------------------------------------------------------

FIELD_LABEL = "field-kuramoto-torus-random-lam0.3-clamp1"


def arm_versus(cells: list[Cell], a_label: str, b_label: str, task: str = "recognition", tier: str = "tier1",
               b_uses_gain: bool = True, floor_span: str = "@wholeclip", width=PRIMARY_WIDTH) -> dict:
    """a minus b at each discriminating condition, paired by seed (and pair, for the order task)."""
    prim = [c for c in primary(cells, task, width=width, floor_span=floor_span) if c.tier == tier]
    idx = _index(prim, lambda c: (c.label, c.noise, c.gain, c.seed, c.pair))
    out = {}
    for noise, gain in conditions():
        pairs = []
        for seed in SEEDS:
            for pair in ({c.pair for c in prim} or {()}):
                a = idx.get((a_label, noise, gain, seed, pair))
                b = idx.get((b_label, noise, gain if b_uses_gain else None, seed, pair))
                if a and b:
                    pairs.append((a, b))
        out[f"{noise:g}db-g{gain:g}"] = compare(pairs)
    return out


def hypotheses_h1_to_h3(cells: list[Cell]) -> dict:
    out = {}
    tests = {"H1": ("floor", False, "H1"), "H2/bankA": ("bank-c4", True, "H2"),
             "H2/bankB": ("bank-c8", True, "H2"), "H3": ("severed-kuramoto-torus-random-lam0.3-clamp1", True, "H3")}
    for name, (other, uses_gain, bar_key) in tests.items():
        per = arm_versus(cells, FIELD_LABEL, other, b_uses_gain=uses_gain)
        outcomes = {k: meets(v, BARS[bar_key]) for k, v in per.items() if v.get("n_pairs")}
        out[name] = {"bar": BARS[bar_key], "conditions": per, "verdict": verdict(outcomes)}
    # the floor over the arms' own window, reported beside H1
    out["H1/armwindow"] = {"conditions": arm_versus(cells, FIELD_LABEL, "floor", b_uses_gain=False,
                                                    floor_span="")}
    return out


def hypothesis_h5(cells: list[Cell]) -> dict:
    """Width 4,096 minus width 192, per arm, at the primary size."""
    out = {}
    base = [c for c in cells if c.tier == "tier1" and c.task == "recognition" and c.n_train == PRIMARY_SIZE]
    labels = sorted({c.label for c in base})
    for label in labels:
        mine = [c for c in base if c.label == label]
        read = "windowed@wholeclip" if mine[0].arm["kind"] == "floor" else "windowed"
        wide = _index([c for c in mine if c.read == read and c.width == 4096], lambda c: (c.noise, c.gain, c.seed))
        narrow = _index([c for c in mine if c.read == read and c.width == 192], lambda c: (c.noise, c.gain, c.seed))
        per = {}
        gain_applies = mine[0].gain is not None
        for noise, gain in conditions(gain_applies):
            pairs = [(wide[k], narrow[k]) for k in [(noise, gain, s) for s in SEEDS] if k in wide and k in narrow]
            per[f"{noise:g}db-g{gain:g}" if gain is not None else f"{noise:g}db"] = compare(pairs)
        outcomes = {k: meets(v, BARS["H5"]) for k, v in per.items() if v.get("n_pairs")}
        out[label] = {"bar": BARS["H5"], "conditions": per, "verdict": verdict(outcomes)}
    return out


def order_gate(cells: list[Cell]) -> dict:
    """A pair is valid at a noise level if both floor reads' intervals contain chance."""
    floors = [c for c in cells if c.tier == "tier1" and c.task == "order" and c.arm["kind"] == "floor"
              and c.width == PRIMARY_WIDTH and c.n_train == PRIMARY_SIZE]
    out = {}
    for pair in sorted({c.pair for c in floors}):
        for noise in sorted({c.noise for c in floors}, key=lambda x: -1 if x is None else x):
            ok = True
            detail = {}
            for read in ("pooled@wholeclip", "pooled"):
                mine = [c for c in floors if c.pair == pair and c.noise == noise and c.read == read]
                if len(mine) < len(SEEDS):
                    ok = False
                    continue
                per_clip = np.mean([_bits(c) for c in mine], axis=0)
                rng = np.random.default_rng(BOOT_SEED)
                boots = per_clip[rng.integers(0, len(per_clip), (BOOTSTRAP, len(per_clip)))].mean(axis=1)
                lo, hi = np.quantile(boots, [(1 - CONFIDENCE) / 2, (1 + CONFIDENCE) / 2])
                detail[read] = {"acc": float(per_clip.mean()), "ci": (float(lo), float(hi))}
                ok &= bool(lo <= 0.5 <= hi)
            out[f"pair{pair[0]}{pair[1]}/{'clean' if noise is None else f'{noise:g}db'}"] = {"valid": ok, **detail}
    return out


def hypothesis_h6(cells: list[Cell], gate: dict) -> dict:
    prim = [c for c in primary(cells, "order") if c.tier == "tier1"]
    field = _index([c for c in prim if c.label == FIELD_LABEL], lambda c: (c.pair, c.noise, c.gain, c.seed))
    out = {"accuracy": {}, "versus_bank_a": {}}
    outcomes_acc, outcomes_diff = {}, {}
    for noise, gain in conditions():
        valid = [p for p in sorted({c.pair for c in prim})
                 if gate.get(f"pair{p[0]}{p[1]}/{noise:g}db", {}).get("valid")]
        passing = 0
        per_pair = {}
        for p in valid:
            accs = [field[k].acc for k in [(p, noise, gain, s) for s in SEEDS] if k in field]
            if len(accs) == len(SEEDS):
                per_pair[f"{p[0]}{p[1]}"] = statistics.fmean(accs)
                passing += statistics.fmean(accs) >= BARS["H6_acc"]
        key = f"{noise:g}db-g{gain:g}"
        out["accuracy"][key] = {"valid_pairs": len(valid), "passing": passing, "per_pair": per_pair}
        if valid:
            outcomes_acc[key] = passing >= BARS["H6_pairs"]
        bank = arm_versus([c for c in cells if c.pair in valid or c.task != "order"], FIELD_LABEL, "bank-c4",
                          task="order")[key]
        out["versus_bank_a"][key] = bank
        if bank.get("n_pairs"):
            outcomes_diff[key] = meets(bank, BARS["H6_diff"])
    out["verdict_accuracy"] = verdict(outcomes_acc)
    out["verdict_versus_bank_a"] = verdict(outcomes_diff)
    return out


def hypothesis_h4(cells: list[Cell]) -> dict:
    """Each design factor against its reference level, over twins, at each discriminating condition."""
    prim = [c for c in primary(cells) if c.tier == "tier2"]
    factors = ("boundary", "omega", "physics", "damping", "clamp")

    def key(c: Cell, skip: str) -> tuple:
        return (tuple((f, c.arm[f]) for f in factors if f != skip), c.noise, c.gain, c.seed)

    out = {}
    for factor in factors:
        levels = sorted({c.arm[factor] for c in prim}, key=str)
        if not levels:                               # tier 2 not run yet
            continue
        ref = REFERENCE.get(factor, levels[0])
        refs = _index([c for c in prim if c.arm[factor] == ref], lambda c: key(c, factor))
        bar = {"boundary": BARS["shape"], "omega": BARS["omega"], "physics": BARS["family"]}.get(factor)
        out[factor] = {"reference": ref, "bar": bar, "levels": {}}
        for level in levels:
            if level == ref:
                continue
            per = {}
            for noise, gain in conditions():
                pairs = [(c, refs[key(c, factor)]) for c in prim
                         if c.arm[factor] == level and c.noise == noise and c.gain == gain and key(c, factor) in refs]
                per[f"{noise:g}db-g{gain:g}"] = compare(pairs)
            entry = {"conditions": per}
            if bar is not None:
                signed = -bar if (factor == "physics" and statistics.fmean(
                    [v["diff"] for v in per.values() if v.get("n_pairs")] or [0]) < 0) else bar
                entry["verdict"] = verdict({k: meets(v, signed) for k, v in per.items() if v.get("n_pairs")})
            out[factor]["levels"][str(level)] = entry
    return out


def drives(cells: list[Cell]) -> dict:
    """A drive over its own floor (H4, +5), for the quadrature and carrier diagonals."""
    out = {}
    for drive in ("quadrature", "carrier"):
        prim = [c for c in primary(cells) if c.tier == "tier3" and c.drive == drive]
        floor = _index([c for c in prim if c.arm["kind"] == "floor"], lambda c: (c.noise, c.seed))
        per = {}
        for label in sorted({c.label for c in prim if c.arm["kind"] == "field"}):
            for noise in sorted({c.noise for c in prim if c.noise is not None}):
                for gain in sorted({c.gain for c in prim if c.gain is not None}):
                    pairs = [(c, floor[(noise, c.seed)]) for c in prim if c.label == label and c.noise == noise
                             and c.gain == gain and (noise, c.seed) in floor]
                    if pairs:
                        per[f"{label}/{noise:g}db-g{gain:g}"] = compare(pairs)
        out[drive] = {"bar": BARS["drive"], "cells": per,
                      "met": sorted(k for k, v in per.items() if meets(v, BARS["drive"]))}
    return out


def table(cells: list[Cell]) -> dict:
    """Three-seed mean accuracy per arm, condition, read, width and size: the curves."""
    groups = defaultdict(list)
    for c in cells:
        groups[(c.tier, c.task, c.drive, c.label, c.noise, c.gain, c.pair, c.read, str(c.width), c.n_train)].append(c.acc)
    return {"/".join(map(str, k)): {"mean": statistics.fmean(v), "n": len(v)} for k, v in sorted(groups.items(), key=str)}


def score() -> dict:
    cells = load()
    gate = order_gate(cells)
    return {"H1-H3": hypotheses_h1_to_h3(cells), "H4": hypothesis_h4(cells), "H4/drives": drives(cells),
            "H5": hypothesis_h5(cells), "order_gate": gate, "H6": hypothesis_h6(cells, gate),
            "table": table(cells)}


def _fmt(result: dict) -> str:
    if not result.get("n_pairs"):
        return "not run"
    seeds = " ".join(f"{v:+.1f}" for v in result["seeds"].values())
    return f"{result['diff']:+6.2f} [{result['ci'][0]:+.2f}, {result['ci'][1]:+.2f}]  seeds {seeds}"


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    out = score()
    for name, entry in out["H1-H3"].items():
        print(f"\n{name}  bar {entry.get('bar', '-')}  -> {entry.get('verdict', 'reported')}")
        for cond, r in entry["conditions"].items():
            print(f"    {cond:10s} {_fmt(r)}")
    print("\nH5, width 4,096 minus 192")
    for label, entry in out["H5"].items():
        print(f"  {label:48s} -> {entry['verdict']}")
    print("\nOrder-task gate")
    for k, v in out["order_gate"].items():
        print(f"  {k:18s} {'valid' if v['valid'] else 'INVALID'}")
    print(f"\nH6 accuracy -> {out['H6']['verdict_accuracy']}; versus bank A -> {out['H6']['verdict_versus_bank_a']}")
    for factor, entry in out["H4"].items():
        print(f"\nH4 {factor} (reference {entry['reference']}, bar {entry['bar']})")
        for level, e in entry["levels"].items():
            print(f"  {level:10s} -> {e.get('verdict', 'no bar')}")
    path = rn.record_root() / "verdicts.json"
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

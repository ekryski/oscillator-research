"""Every accuracy and every comparison in the record, with its spread.

    uv run python -m harness.confirm.summary      # write summary.json and summary.md

No threshold decides anything here. An accuracy is reported as its mean over
replicates (three seeds under Protocol A, five folds under Protocol B), the
sample standard deviation and each replicate's own value, all in points. A
comparison A minus B is paired: matched on condition and replicate, it reports
each replicate's difference, their mean and standard deviation, and a 95%
interval from resampling test clips. Seeds share one test set, so their
per-clip differences are averaged; folds and order-task pairs each have their
own test clips, so theirs are concatenated. Whether a difference is a gain is
left to the reader.

Nothing decides whether a result passes; DESIGN.md's decision log records
when the design's first pass/fail bars were dropped.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

import numpy as np

from harness.confirm import plan, terms
from harness.confirm import record as rec
from harness.confirm import run as rn
from harness.confirm.record import Cell

FIELD = rec.FIELD_LABEL
SEVERED = FIELD.replace("field-", "severed-", 1)
READ = {"recognition": "windowed", "order": "pooled"}
#: the reference level of each design factor: the Tier 1 network's own
REFERENCE = terms.REFERENCE
#: bootstrap resamples drawn per batch, bounding memory on Protocol B's 30,000 test clips
CHUNK = 100


def replicate(c: Cell) -> str:
    return f"fold{c.fold}" if c.fold >= 0 else f"seed{c.seed}"


def spread(values: dict[str, float]) -> dict:
    """Mean, sample standard deviation and each replicate's value, from fractions to points."""
    pts = {k: 100 * v for k, v in sorted(values.items())}
    xs = list(pts.values())
    return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else None,
            "n": len(xs), "values": pts}


def interval(per_clip: np.ndarray) -> list[float]:
    """The 95% interval of the mean from resampling clips, in points."""
    rng = np.random.default_rng(rec.BOOT_SEED)
    boots = np.concatenate([per_clip[rng.integers(0, len(per_clip), (CHUNK, len(per_clip)))].mean(axis=1)
                            for _ in range(rec.BOOTSTRAP // CHUNK)])
    lo, hi = np.quantile(boots, [(1 - rec.CONFIDENCE) / 2, (1 + rec.CONFIDENCE) / 2])
    return [100 * float(lo), 100 * float(hi)]


def paired(pairs: list[tuple[Cell, Cell]]) -> dict:
    """A minus B over matched cells.

    A replicate's difference is the mean over its own pairs (matched pairs,
    order-task digit pairs). The interval resamples test clips: per-clip differences are averaged
    over cells that share a test set and concatenated over test sets that do not.
    """
    by_rep = defaultdict(list)
    for a, b in pairs:
        by_rep[replicate(a)].append(a.acc - b.acc)
    out = {**spread({r: statistics.fmean(v) for r, v in by_rep.items()}), "n_pairs": len(pairs)}
    if all(a.bits and b.bits for a, b in pairs):
        by_set = defaultdict(list)
        for a, b in pairs:
            by_set[(a.fold, a.pair)].append(rec.bits(a) - rec.bits(b))
        out["ci95"] = interval(np.concatenate([np.mean(v, axis=0) for _, v in sorted(by_set.items())]))
    return out


# ---------------------------------------------------------------------------
# Accuracies
# ---------------------------------------------------------------------------

def accuracies(cells: list[Cell]) -> list[dict]:
    """One record per arm, read, condition, width and size, over its replicates."""
    groups = defaultdict(dict)
    for c in cells:
        key = (c.tier, c.task, c.drive, c.label, c.read, c.noise, c.gain, c.pair, c.width, c.n_train, c.projection)
        groups[key][replicate(c)] = c.acc
    fields = ("tier", "task", "drive", "arm", "read", "noise", "gain", "pair", "width", "n_train", "projection")
    return [{**dict(zip(fields, k, strict=True)), "name": terms.arm(k[3]), "pathway": terms.PATHWAYS[k[2]],
             "pair": list(k[7]) or None, **spread(v)} for k, v in sorted(groups.items(), key=str)]


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def matched(a_cells: list[Cell], b_cells: list[Cell], widths: tuple | None = None) -> dict[tuple, list]:
    """Pairs (a, b) matched on condition, replicate, width and size, grouped by condition, width and size.

    `widths=(wa, wb)` takes a at width wa and b at width wb rather than width for
    width. An arm without gain is matched at every gain of the other.
    Order-task pairs are pooled into one group per condition.
    """
    b_gain = any(c.gain is not None for c in b_cells)
    idx = {(c.noise, c.gain, c.pair, c.width, c.n_train, replicate(c)): c for c in b_cells}
    groups = defaultdict(list)
    for c in a_cells:
        if widths and c.width != widths[0]:
            continue
        other = idx.get((c.noise, c.gain if b_gain else None, c.pair, widths[1] if widths else c.width,
                         c.n_train, replicate(c)))
        if other is not None:
            groups[(c.noise, c.gain, c.width, c.n_train)].append((c, other))
    return groups


def versus(cells: list[Cell], name: str, a: tuple[str, str], b: tuple[str, str], *, tier: str, task: str,
           drive: str = "envelope", widths: tuple | None = None, projection: str = "fixed") -> list[dict]:
    """a minus b, each an (arm, read), in one tier and under one projection."""
    mine = [c for c in cells if c.tier == tier and c.task == task and c.drive == drive and c.projection == projection]
    groups = matched([c for c in mine if (c.label, c.read) == a], [c for c in mine if (c.label, c.read) == b], widths)
    return [{"comparison": name, "a": " ".join(a), "b": " ".join(b), "tier": tier, "task": task, "drive": drive,
             "noise": k[0], "gain": k[1], "width": k[2], "b_width": widths[1] if widths else k[2],
             "n_train": k[3], "projection": projection, **paired(v)} for k, v in sorted(groups.items(), key=str)]


BASELINE_WHOLE = "the spectrogram-only baseline, whole clip"
BASELINE_16 = "the spectrogram-only baseline, from frame 16"
NETWORK = terms.arm(FIELD)


def _against_network(cells: list[Cell], tier: str, task: str) -> list[dict]:
    r = READ[task]
    others = [(BASELINE_WHOLE, ("floor", r + "@wholeclip")), (BASELINE_16, ("floor", r)),
              ("the " + terms.arm(SEVERED), (SEVERED, r)), ("the " + terms.arm("bank-c4"), ("bank-c4", r)),
              ("the " + terms.arm("bank-c8"), ("bank-c8", r))]
    others += [(f"the {terms.arm('ann-' + arch)}", (f"ann-{arch}", r)) for arch in plan.ANN_ARCHS]
    out = []
    for name, b in others:
        out += versus(cells, f"{NETWORK} minus {name}", (FIELD, r), b, tier=tier, task=task)
    return out


def tier1(cells: list[Cell]) -> list[dict]:
    """The coupled network against every other arm, rotation rates, and width 4,096 against 192 per arm."""
    out = []
    for task in ("recognition", "order"):
        r = READ[task]
        out += _against_network(cells, "tier1", task)
        out += versus(cells, f"{NETWORK}, with rotation rates minus without", (FIELD, r + "+rate"), (FIELD, r),
                      tier="tier1", task=task)
        for label in sorted({c.label for c in cells if c.tier == "tier1" and c.task == task}):
            read = r + "@wholeclip" if label == "floor" else r
            name = terms.arm(label) + (", whole clip" if label == "floor" else "")
            out += versus(cells, f"{name}: width 4,096 minus 192", (label, read), (label, read),
                          tier="tier1", task=task, widths=(4096, 192))
    return out


def design(cells: list[Cell]) -> list[dict]:
    """Each Tier 2 factor level minus its reference, over matched pairs identical in every other factor."""
    prim = [c for c in cells if c.tier == "tier2" and c.read == "windowed"]
    out = []
    for factor, ref in REFERENCE.items():
        def match(c: Cell, factor: str = factor) -> tuple:
            return (tuple(c.arm[f] for f in REFERENCE if f != factor), c.noise, c.gain, c.width, c.n_train,
                    replicate(c))
        refs = {match(c): c for c in prim if c.arm[factor] == ref}
        for level in sorted({c.arm[factor] for c in prim} - {ref}, key=str):
            groups = defaultdict(list)
            for c in prim:
                if c.arm[factor] == level and match(c) in refs:
                    groups[(c.noise, c.gain, c.width, c.n_train)].append((c, refs[match(c)]))
            name = f"{terms.FACTORS[factor]}: {terms.level(level)} minus {terms.level(ref)}"
            out += [{"comparison": name, "tier": "tier2", "task": "recognition",
                     "drive": "envelope", "noise": k[0], "gain": k[1], "width": k[2], "b_width": k[2],
                     "n_train": k[3], **paired(v)} for k, v in sorted(groups.items(), key=str)]
    # rotation rates, over every design configuration
    idx = {(c.label, c.noise, c.gain, c.width, c.n_train, replicate(c)): c for c in prim}
    groups = defaultdict(list)
    for c in cells:
        if c.tier == "tier2" and c.read == "windowed+rate":
            other = idx.get((c.label, c.noise, c.gain, c.width, c.n_train, replicate(c)))
            if other is not None:
                groups[(c.noise, c.gain, c.width, c.n_train)].append((c, other))
    out += [{"comparison": "with rotation rates minus without, every design configuration", "tier": "tier2",
             "task": "recognition",
             "drive": "envelope", "noise": k[0], "gain": k[1], "width": k[2], "b_width": k[2], "n_train": k[3],
             **paired(v)} for k, v in sorted(groups.items(), key=str)]
    return out


def drives(cells: list[Cell]) -> list[dict]:
    """Every arm on the quadrature and carrier pathways, minus that pathway's own spectrogram-only baseline."""
    out = []
    for drive in ("quadrature", "carrier"):
        for label in sorted({c.label for c in cells if c.tier == "tier3" and c.drive == drive} - {"floor"}):
            for name, read in ((BASELINE_WHOLE, "windowed@wholeclip"), (BASELINE_16, "windowed")):
                out += versus(cells, f"{terms.arm(label)} minus {name}", (label, "windowed"), ("floor", read),
                              tier="tier3", task="recognition", drive=drive)
    return out


def projections(cells: list[Cell]) -> list[dict]:
    """The projection tier: each reservoir under the seeded projection minus the fixed one, and the coupled
    network against its controls under the seeded projection (the spectrogram-only baseline and the trained
    baselines are never projected, so their Tier 1 cells stand for both)."""
    out = []
    for task in ("recognition", "order"):
        r = READ[task]
        mine = [c for c in cells if c.tier == "projection" and c.task == task and c.read == r]
        seeded = [c for c in mine if c.projection == "seeded"]
        for label in sorted({c.label for c in mine}):
            groups = matched([c for c in seeded if c.label == label],
                             [c for c in mine if c.label == label and c.projection == "fixed"])
            out += [{"comparison": f"{terms.arm(label)}: seeded minus fixed projection", "tier": "projection",
                     "task": task, "noise": k[0], "gain": k[1], "width": k[2], "n_train": k[3],
                     **paired(v)} for k, v in sorted(groups.items(), key=str)]
        network = [c for c in seeded if c.label == FIELD]
        controls = [(BASELINE_WHOLE, [c for c in cells if c.tier == "tier1" and c.task == task and c.label == "floor"
                                      and c.read == r + "@wholeclip"])]
        controls += [("the " + terms.arm(lab), [c for c in seeded if c.label == lab]) for lab in (SEVERED, "bank-c4", "bank-c8")]
        for name, b_cells in controls:
            groups = matched(network, b_cells)
            out += [{"comparison": f"{NETWORK} minus {name}, seeded projection", "tier": "projection", "task": task,
                     "noise": k[0], "gain": k[1], "width": k[2], "n_train": k[3], "projection": "seeded",
                     **paired(v)} for k, v in sorted(groups.items(), key=str)]
    return out


def summary(cells: list[Cell] | None = None) -> dict:
    cells = rec.load() if cells is None else cells
    return {"accuracy": accuracies(cells),
            "comparisons": (tier1(cells) + _against_network(cells, "becker", "recognition") + design(cells)
                            + drives(cells) + projections(cells)),
            "order_baseline_at_chance": rec.baseline_at_chance(cells)}


# ---------------------------------------------------------------------------
# The readable report: the primary cells
# ---------------------------------------------------------------------------

def _noise(n) -> str:
    return "clean" if n is None else f"{n:+g} dB".replace("+0 dB", "0 dB")


def _by_noise(r: dict) -> tuple:
    return (-1 if r["noise"] is None else r["noise"],), _noise(r["noise"])


def _by_condition(r: dict) -> tuple:
    gain = f", gain = {r['gain']:g}" if r["gain"] is not None else ""
    return (-1 if r["noise"] is None else r["noise"], r["gain"] or 0), _noise(r["noise"]) + gain


def _by_size(r: dict) -> tuple:
    return (r["n_train"], r["width"]), f"n {r['n_train']:,}, w {r['width']:,}"


def _acc(r: dict) -> str:
    return f"{r['mean']:.1f}" + (f" ± {r['sd']:.1f}" if r["sd"] is not None else "")


def _diff(r: dict) -> str:
    s = f"{r['mean']:+.2f}" + (f" ± {r['sd']:.2f}" if r["sd"] is not None else "")
    return s + (f" [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]" if "ci95" in r else "")


def _grid(records: list[dict], row, col, cell) -> list[str]:
    """A table: one row per `row(r)`, one column per `col(r)`, a (sort key, heading) pair."""
    rows: dict = defaultdict(dict)
    for r in records:
        rows[row(r)][col(r)] = cell(r)
    if not rows:
        return ["(not run yet)"]
    cols = sorted({c for v in rows.values() for c in v})
    out = ["| | " + " | ".join(h for _, h in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    return out + [f"| {k} | " + " | ".join(v.get(c, "") for c in cols) + " |" for k, v in rows.items()]


def _pooled(records: list[dict]) -> list[dict]:
    """Order-task accuracies averaged over pairs within each replicate."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        for rep, v in r["values"].items():
            groups[(r["arm"], r["read"], r["noise"], r["gain"])][rep].append(v / 100)
    return [{"arm": k[0], "read": k[1], "noise": k[2], "gain": k[3],
             **spread({rep: statistics.fmean(v) for rep, v in reps.items()})} for k, reps in groups.items()]


#: the order arms are listed in: the baseline, the reservoirs, then the trained baselines
ORDER = ["floor", FIELD, SEVERED, "bank-c4", "bank-c8"] + [f"ann-{a}" for a in ("transformer", "gru", "s4d", "cnn", "tcn")]


def _rank(r: dict) -> tuple:
    return (ORDER.index(r["arm"]) if r["arm"] in ORDER else len(ORDER), not r["read"].endswith("@wholeclip"),
            r.get("gain") or 0)


def _pooled_projection(records: list[dict]) -> list[dict]:
    """Order-task accuracies averaged over pairs, kept apart by projection."""
    out = []
    for proj in ("fixed", "seeded"):
        out += [{**r, "projection": proj} for r in _pooled([r for r in records if r["projection"] == proj])]
    return out


def _arm(r: dict) -> str:
    name = terms.arm(r["arm"])
    if r["arm"] == "floor":
        name += ", whole clip" if r["read"].endswith("@wholeclip") else ", from frame 16"
    return name + (f" (gain = {r['gain']:g})" if r.get("gain") is not None else "")


def progress() -> dict[str, tuple[int, int]]:
    """Runs recorded and runs planned, per tier."""
    root, have, out = rn.record_root(), {}, {}
    for name, tier in plan.TIERS.items():
        specs = list(tier())
        for s in specs:
            if s.group() not in have:
                path = root / f"{s.group()}.json"
                have[s.group()] = set(json.loads(path.read_text())["runs"]) if path.exists() else set()
        out[name] = (sum(s.run_id() in have[s.group()] for s in specs), len(specs))
    return out


def report(s: dict, done: dict[str, tuple[int, int]]) -> str:
    acc, cmp = s["accuracy"], s["comparisons"]
    w, n = rec.PRIMARY_WIDTH, rec.PRIMARY_SIZE

    def prim_acc(tier, task, n_train=n):
        return sorted((r for r in acc if r["tier"] == tier and r["task"] == task and r["width"] == w
                       and r.get("projection", "fixed") == "fixed"
                       and r["n_train"] == n_train and r["read"].split("@")[0] == READ[task]), key=_rank)

    def prim_cmp(tier, n_train=n, prefix=""):
        return [r for r in cmp if r["tier"] == tier and r["width"] == w and r["n_train"] == n_train
                and r["comparison"].startswith(prefix) and "width" not in r["comparison"]]

    def by_gain(r):
        return r["comparison"] + (f" (gain = {r['gain']:g})" if r["gain"] is not None else "")

    lines = ["# Confirmatory results", "",
             "Generated by `uv run python -m harness.confirm.summary` from the record in this folder; every number "
             "here and more (every width, size, pair and read) is in `summary.json`. Accuracies are mean ± "
             "standard deviation over replicates (three seeds; five folds in Tier B), in percent. Differences "
             "are paired, mean ± standard deviation over replicates, with the 95% interval from resampling test "
             f"clips in brackets, in points. Primary cell: width {w}, {n:,} training clips.", "",
             "Runs recorded: " + "; ".join(f"{k} {d:,} of {p:,}" for k, (d, p) in done.items())
             + ". A tier not yet complete is summarized over the runs it has.", ""]
    lines += ["## Tier 1, recognition: accuracy", ""]
    lines += _grid(prim_acc("tier1", "recognition"), _arm, _by_noise, _acc)
    lines += ["", "## Tier 1, recognition: differences", ""]
    lines += _grid([r for r in prim_cmp("tier1") if r["task"] == "recognition"], by_gain,
                   _by_noise, _diff)
    lines += ["", "## Tier 1, recognition: accuracy by width and training size", ""]
    for noise in (0.0, 5.0):
        rows = [r for r in acc if r["tier"] == "tier1" and r["task"] == "recognition" and r["noise"] == noise
                and r["read"] in ("windowed", "windowed@wholeclip") and r["width"] != "native"]
        rows.sort(key=_rank)
        lines += [f"### {_noise(noise)}", ""]
        lines += _grid(rows, _arm, _by_size, _acc) + [""]
    lines += ["## Tier 1, recognition: width 4,096 minus 192", ""]
    lines += _grid([r for r in cmp if r["tier"] == "tier1" and r["task"] == "recognition" and r["n_train"] == n
                    and "width" in r["comparison"]], by_gain, _by_noise, _diff)
    lines += ["", "## Tier 1, order task: accuracy, averaged over the five pairs", ""]
    lines += _grid(sorted(_pooled(prim_acc("tier1", "order")), key=_rank), _arm, _by_noise, _acc)
    lines += ["", "## Tier 1, order task: differences, pooled over the five pairs", ""]
    lines += _grid([r for r in prim_cmp("tier1") if r["task"] == "order"], by_gain,
                   _by_noise, _diff)
    at = s["order_baseline_at_chance"]
    off = [k for k, v in at.items() if not all(r["contains_chance"] for r in v.values())]
    lines += ["", "Order task, the spectrogram-only baseline's 95% interval contains chance (50%) in "
              f"{len(at) - len(off)} of {len(at)} pair and noise cells"
              + (f"; it does not in: {', '.join(off)}." if off else "."), ""]
    lines += ["## Tier 2, design: each level minus its reference, over matched pairs", ""]
    lines += _grid(prim_cmp("tier2"), lambda r: r["comparison"], _by_condition, _diff)
    proj = [r for r in acc if r["tier"] == "projection" and r["width"] == w and r["n_train"] == n]
    lines += ["", "## Tier 1's reservoirs under the fixed and the seeded projection", ""]
    if proj:
        base = {(r["task"], r["arm"], r["read"], r["noise"], r["gain"], str(r["pair"]), r["width"], r["n_train"]): r
                for r in acc if r["tier"] == "tier1"}
        drift = [abs(r["mean"] - base[k]["mean"]) for r in proj if r["projection"] == "fixed"
                 and (k := (r["task"], r["arm"], r["read"], r["noise"], r["gain"], str(r["pair"]), r["width"],
                            r["n_train"])) in base]
        lines += [f"The fixed cells reproduce Tier 1's: largest difference {max(drift, default=0):.4f} points "
                  f"over {len(drift)} cells.", ""]
    for task in ("recognition", "order"):
        rows = [r for r in proj if r["task"] == task]
        rows = _pooled_projection(rows) if task == "order" else rows
        lines += [f"### {task}", ""]
        lines += _grid(sorted(rows, key=_rank), lambda r: f"{_arm(r)}, {r['projection']}", _by_noise, _acc) + [""]
    lines += _grid([r for r in cmp if r["tier"] == "projection" and r["width"] == w and r["n_train"] == n],
                   lambda r: f"{r['task']}: {r['comparison']}" + (f" (gain = {r['gain']:g})" if r["gain"] is not None else ""),
                   _by_noise, _diff)
    lines += ["", "## Tier B, Becker et al.'s folds: accuracy", ""]
    lines += _grid(prim_acc("becker", "recognition", plan.BECKER_TRAIN), _arm, _by_noise, _acc)
    lines += ["", "## Tier B: differences", ""]
    lines += _grid(prim_cmp("becker", plan.BECKER_TRAIN), by_gain, _by_noise, _diff)
    lines += ["", "## Tier 3, quadrature and carrier pathways: each arm minus that pathway's baseline", ""]
    lines += _grid([r for r in prim_cmp("tier3")], lambda r: f"{terms.PATHWAYS[r['drive']]} pathway: {r['comparison']}",
                   _by_condition, _diff)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    s = summary()
    root = rn.record_root()
    (root / "summary.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    text = report(s, progress())
    (root / "summary.md").write_text(text)
    print(text)
    print(f"wrote {root / 'summary.json'} and {root / 'summary.md'}")


if __name__ == "__main__":
    main()

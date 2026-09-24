"""Every accuracy and every comparison in paper 03's record, with its spread.

    uv run python -m harness.confirm.summary      # write summary.json and summary.md

Reported as paper 02 reports: an
accuracy is its mean over the three seeds, the sample standard deviation and
each seed's value, in points; a comparison A minus B is paired, matched on
lattice, band mapping, channel count, noise level, input gain and seed, and
reported as each seed's difference, their mean and standard deviation, and a
95% interval from resampling test clips. No threshold is applied and no
verdict drawn.

Every projected read is reported twice: under the fixed projection (paper
02's matrix, the same for every seed) and under the projection seeded by the
run's seed, whose spread over seeds includes the projection's own draw. An
unprojected read (a width at or above the arm's own) is tagged "none" and is
compared with either. Paper 02's cells recorded before it tagged projections
are its fixed ones.

Cells taken from paper 02 (16 x 16, 4 channels; `plan.reused`) are read from
paper 02's record and reported under paper 03's tier, marked with their
source.

The memory tasks add cells of their own, derived from the recorded ones and
never recorded: on the order task, the five digit pairs pooled ("all" pairs,
their test clips taken together); on the digit-sequence task, the mean over
positions ("mean": each clip's share of positions right) and every position
right at once ("all"). Their intervals resample the clips.

A network's instruments (how synchronized it is, and how locked to its drive;
harness.confirm.arms.field_instruments) are reported beside its accuracies,
as the mean over its test clips, then over seeds. They are diagnostics: no
cell depends on them.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, replace

import numpy as np

from harness.confirm import arms as am
from harness.confirm import plan, terms
from harness.confirm import protocol as pr
from harness.confirm import run as rn
from harness.confirm.readout import projection_of, unpack
from harness.measurement.features import PROJECTIONS

PRIMARY_WIDTH, PRIMARY_SIZE = 192, rn.PRIMARY_SIZE
READ = "windowed"
#: what the instruments record, and their names in the report
INSTRUMENTS = {"R": "order parameter R", "plv": "phase locking to the drive", "entrained": "share entrained",
               "amplitude": "amplitude"}
BOOTSTRAP, CONFIDENCE, BOOT_SEED = 2000, 0.95, 20260923
#: bootstrap resamples drawn per batch, bounding memory
CHUNK = 100
#: record files that are not runs
NOT_RUNS = ("gates.json", "summary.json")


@dataclass(frozen=True)
class Cell:
    """One recorded accuracy, with what it belongs to."""
    tier: str
    drive: str
    noise: float | None
    gain: float | None
    seed: int
    arm: dict
    label: str
    read: str
    width: object
    n_train: int
    acc: float
    bits: str | None
    n_test: int
    source: str = "paper 03"
    projection: str = "fixed"
    task: str = "recognition"
    pair: object = ()              # the order task's pair, or "all" for the pairs pooled
    length: int = 0                # the digit-sequence task's length
    position: object = None        # a sequence position, or "mean" / "all" over positions
    scores: np.ndarray | None = field(default=None, compare=False, repr=False)   # a derived cell's per clip

    @property
    def grid(self) -> int:
        return self.arm.get("grid", 16)

    @property
    def bands(self) -> int:
        """The band mapping: 0 for one band per row (always 0 at 16 x 16)."""
        b = self.arm.get("bands", 0)
        return 0 if b in (0, self.grid) else b

    @property
    def window(self) -> int:
        """The analysis window: 0 for paper 02's 512 samples."""
        w = self.arm.get("window", 0)
        return 0 if w in (0, am.WINDOW) else w

    @property
    def channels(self) -> int:
        return self.arm.get("channels", 4)

    @property
    def size(self) -> tuple[int, int, int, int]:
        return self.grid, self.bands, self.window, self.channels

    @property
    def condition(self) -> tuple:
        """The task and its set: the order task's pair, the sequence's length and position."""
        return self.task, self.pair, self.length, self.position


@dataclass(frozen=True)
class Instruments:
    """One network run's instruments, over its test clips."""
    tier: str
    task: str
    pair: tuple
    length: int
    drive: str
    noise: float | None
    gain: float | None
    seed: int
    arm: dict
    label: str
    values: dict
    source: str


def _label(run_id: str, kind: str) -> str:
    return run_id.split("/")[-1] if kind != "ann" else run_id.split("/")[-2]


def _cells(tier: str, rec: dict, run_id: str, source: str) -> list[Cell]:
    s = rec["spec"]
    label = _label(run_id, s["arm"]["kind"])
    return [Cell(tier, s["drive"], s["noise_db"], s["gain"], s["seed"], s["arm"], label, c["read"], c["width"],
                 c["n_train"], c["acc"], c.get("correct"), rec["n_test"], source, projection_of(c, rec),
                 s.get("task", "recognition"), tuple(s.get("pair", ())), s.get("length", 0), c.get("position"))
            for c in rec["cells"]]


def _instruments(tier: str, rec: dict, run_id: str, source: str) -> list[Instruments]:
    s = rec["spec"]
    if "instruments" not in rec:
        return []
    return [Instruments(tier, s.get("task", "recognition"), tuple(s.get("pair", ())), s.get("length", 0),
                        s["drive"], s["noise_db"], s["gain"], s["seed"], s["arm"], _label(run_id, s["arm"]["kind"]),
                        {k: v["mean"] for k, v in rec["instruments"].items()}, source)]


def _scores(c: Cell) -> np.ndarray:
    """Per test clip: 1 or 0 for a recorded cell, and a derived cell's own per-clip values."""
    return c.scores if c.scores is not None else unpack(c.bits, c.n_test).numpy().astype(np.float64)


def _same_read(c: Cell) -> tuple:
    return (c.tier, c.drive, c.noise, c.gain, c.seed, c.label, c.read, c.width, c.n_train, c.projection, c.task,
            c.length, c.source)


def _derived(cells: list[Cell]) -> list[Cell]:
    """The order task's pairs pooled, and the digit-sequence task's positions taken together."""
    out = []
    pairs, positions = defaultdict(list), defaultdict(list)
    for c in cells:
        if c.task == "order" and c.pair:
            pairs[_same_read(c)].append(c)
        elif c.task == "sequence" and isinstance(c.position, int):
            positions[(*_same_read(c), c.pair)].append(c)
    for group in pairs.values():
        if len({c.pair for c in group}) != len(pr.PAIRS):
            continue
        group = sorted(group, key=lambda c: c.pair)
        scores = (np.concatenate([_scores(c) for c in group]) if all(c.bits for c in group) else None)
        out.append(replace(group[0], pair="all", acc=float(np.mean([c.acc for c in group])), bits=None,
                           n_test=sum(c.n_test for c in group), scores=scores))
    for group in positions.values():
        if sorted(c.position for c in group) != list(range(group[0].length)):
            continue
        both = all(c.bits for c in group)
        each = np.stack([_scores(c) for c in group]) if both else None
        out.append(replace(group[0], position="mean", acc=float(np.mean([c.acc for c in group])), bits=None,
                           scores=None if each is None else each.mean(0)))
        if both:
            out.append(replace(group[0], position="all", acc=float(each.min(0).mean()), bits=None,
                               scores=each.min(0)))
    return out


def load() -> tuple[list[Cell], list[Instruments]]:
    """Paper 03's record, and the planned cells paper 02 recorded (unless paper 03 ran them itself), with
    the cells derived from them, and every network run's instruments."""
    out, inst, ours = [], [], set()
    for path in sorted(rn.record_root().glob("*.json")):
        if path.name in NOT_RUNS:
            continue
        for run_id, rec in json.loads(path.read_text())["runs"].items():
            out += _cells(rec["spec"]["tier"], rec, run_id, "paper 03")
            inst += _instruments(rec["spec"]["tier"], rec, run_id, "paper 03")
            ours.add((path.stem, run_id))
    for tier in plan.TIERS.values():
        for spec in tier():
            mine = (spec.group(), spec.run_id()) in ours
            rec = plan.paper02_run(spec) if plan.reused(spec) and not mine else None
            if rec is not None:
                out += _cells(spec.tier, rec, spec.run_id(), "paper 02")
                inst += _instruments(spec.tier, rec, spec.run_id(), "paper 02")
    return out + _derived(out), inst


def spread(values: dict[str, float]) -> dict:
    """Mean, sample standard deviation and each seed's value, from fractions to points."""
    pts = {k: 100 * v for k, v in sorted(values.items())}
    xs = list(pts.values())
    return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else None,
            "n": len(xs), "values": pts}


def interval(per_clip: np.ndarray) -> list[float]:
    """The 95% interval of the mean from resampling clips, in points."""
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.concatenate([per_clip[rng.integers(0, len(per_clip), (CHUNK, len(per_clip)))].mean(axis=1)
                            for _ in range(BOOTSTRAP // CHUNK)])
    lo, hi = np.quantile(boots, [(1 - CONFIDENCE) / 2, (1 + CONFIDENCE) / 2])
    return [100 * float(lo), 100 * float(hi)]


def paired(pairs: list[tuple[Cell, Cell]]) -> dict:
    """A minus B over matched cells: each seed's mean difference, and a clip bootstrap.

    Every cell shares the one test set, so per-clip differences are averaged over pairs.
    """
    by_seed = defaultdict(list)
    for a, b in pairs:
        by_seed[f"seed{a.seed}"].append(a.acc - b.acc)
    out = {**spread({r: statistics.fmean(v) for r, v in by_seed.items()}), "n_pairs": len(pairs)}
    if all((a.bits or a.scores is not None) and (b.bits or b.scores is not None) for a, b in pairs):
        out["ci95"] = interval(np.mean([_scores(a) - _scores(b) for a, b in pairs], axis=0))
    return out


# ---------------------------------------------------------------------------
# Accuracies
# ---------------------------------------------------------------------------

def accuracies(cells: list[Cell]) -> list[dict]:
    """One record per arm, read, condition, width and size, over its seeds."""
    groups, sources = defaultdict(dict), defaultdict(set)
    for c in cells:
        key = (c.tier, c.task, c.pair, c.length, c.position, c.drive, c.label, c.read, c.noise, c.gain, c.width,
               c.n_train, c.projection)
        groups[key][f"seed{c.seed}"] = c.acc
        sources[key].add(c.source)
    fields = ("tier", "task", "pair", "length", "position", "drive", "arm", "read", "noise", "gain", "width",
              "n_train", "projection")
    out = []
    for k, v in sorted(groups.items(), key=str):
        base, channels, grid, bands = terms.split(k[6])
        out.append({**dict(zip(fields, k, strict=True)), "name": terms.arm(k[6], k[0]),
                    "pathway": terms.PATHWAYS[k[5]], "grid": grid, "bands": bands or 0,
                    "window": terms.window_of(k[6]) or 0, "channels": channels, "source": sorted(sources[k]),
                    **spread(v)})
    return out


def instruments(runs: list[Instruments]) -> list[dict]:
    """Each network's instruments per condition, over its seeds (each seed's value the mean over its test
    clips). Values are as recorded, not in points."""
    groups, sources = defaultdict(lambda: defaultdict(dict)), defaultdict(set)
    for r in runs:
        key = (r.tier, r.task, r.pair, r.length, r.drive, r.label, r.noise, r.gain)
        for name, v in r.values.items():
            groups[key][name][f"seed{r.seed}"] = v
        sources[key].add(r.source)
    out = []
    for k, by in sorted(groups.items(), key=str):
        base, channels, grid, bands = terms.split(k[5])
        entry = {"tier": k[0], "task": k[1], "pair": k[2], "length": k[3], "drive": k[4], "arm": k[5],
                 "name": terms.arm(k[5], k[0]), "pathway": terms.PATHWAYS[k[4]], "noise": k[6], "gain": k[7],
                 "grid": grid, "bands": bands or 0, "window": terms.window_of(k[5]) or 0, "channels": channels,
                 "source": sorted(sources[k])}
        for name, values in by.items():
            xs = list(values.values())
            entry[name] = {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else None,
                           "n": len(xs), "values": dict(sorted(values.items()))}
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def _kind(c: Cell) -> str:
    a = c.arm
    if a["kind"] == "field":
        return "severed" if a.get("severed") else "field"
    return a["kind"]


def _is_reference(c: Cell) -> bool:
    a = c.arm
    return (a.get("physics"), a.get("boundary")) == plan.REFERENCE


MATCH = ("grid", "bands", "window", "channels")


def compare(cells: list[Cell], name: str, a, b, *, per_channel: bool = True, ignore: tuple = (),
            extra: dict | None = None) -> list[dict]:
    """a minus b over matched cells: `a` and `b` select cells, matched on task and its set (pair, length,
    position), lattice, band mapping, window (and channel count if `per_channel`), noise, gain, width,
    training size, seed and projection; `ignore` drops some of the size's terms from the match (for
    one band mapping or window against another). A cell without gain (the spectrogram-only baseline, a
    trained baseline) matches the other at every gain, and an unprojected cell of b (projection "none")
    matches a under either projection."""
    bs = [c for c in cells if b(c)]
    b_gain = any(c.gain is not None for c in bs)
    drop = set(ignore) | (set() if per_channel else {"channels"})

    def size(c: Cell) -> tuple:
        return tuple(None if m in drop else getattr(c, m) for m in MATCH)

    def key(c: Cell, projection: str) -> tuple:
        return (c.condition, c.drive, size(c), c.noise, c.gain if b_gain else None, c.width, c.n_train, c.seed,
                projection)
    idx = {key(c, c.projection): c for c in bs}
    groups = defaultdict(list)
    for c in cells:
        if not a(c):
            continue
        other = idx.get(key(c, c.projection)) or idx.get(key(c, "none"))
        if other is not None:
            groups[(c.condition, c.drive, c.size, c.noise, c.gain, c.width, c.n_train, c.projection)].append(
                (c, other))
    out = []
    for k, v in sorted(groups.items(), key=str):
        (task, pair, length, position), drive, (grid, bands, window, channels) = k[0], k[1], k[2]
        out.append({"comparison": name, **(extra or {}), "task": task, "pair": pair, "length": length,
                    "position": position, "pathway": terms.PATHWAYS[drive], "grid": grid, "bands": bands,
                    "window": window, "channels": channels, "noise": k[3], "gain": k[4], "width": k[5],
                    "n_train": k[6], "projection": k[7], **paired(v)})
    return out


def _sel(tier: str, kind: str, read: str | None = None, reference: bool | None = True, drive: str | None = None,
         task: str = "recognition"):
    """Cells of one tier, arm kind and task; the task's primary read unless `read` names another."""
    read = read or am.PRIMARY_READ[task]

    def pick(c: Cell) -> bool:
        return (c.tier == tier and c.task == task and _kind(c) == kind and c.read == read
                and (drive is None or c.drive == drive)
                and (reference is None or kind not in ("field", "severed") or _is_reference(c) == reference))
    return pick


NETWORK = "coupled oscillator network"
UNCOUPLED = "uncoupled oscillator network"
BANK = "state-matched leaky-integrator bank"
WHOLE = "spectrogram-only baseline, whole clip"
RATES = "with its rotation rates minus without"


def _controls(cells: list[Cell], tier: str, task: str, drive: str = "envelope", suffix: str = "",
              bank: bool = True) -> list[dict]:
    """The network against the uncoupled network, the bank and the whole-clip baseline, each of those
    against the baseline too, and each network with its rotation rates against without."""
    read = am.PRIMARY_READ[task]
    net, whole = _sel(tier, "field", drive=drive, task=task), _sel(tier, "floor", f"{read}@wholeclip", drive=drive,
                                                                    task=task)
    out = compare(cells, f"{NETWORK} minus the {UNCOUPLED}{suffix}", net, _sel(tier, "severed", drive=drive,
                                                                               task=task))
    kinds = [("field", NETWORK), ("severed", UNCOUPLED)]
    if bank:
        out += compare(cells, f"{NETWORK} minus the {BANK}{suffix}", net, _sel(tier, "bank", drive=drive, task=task))
        kinds.append(("bank", BANK))
    for kind, name in kinds:
        out += compare(cells, f"{name} minus the {WHOLE}{suffix}", _sel(tier, kind, drive=drive, task=task), whole,
                       per_channel=False)
    for kind, name in kinds[:2]:
        out += compare(cells, f"{name} {RATES}{suffix}", _sel(tier, kind, f"{read}+rate", drive=drive, task=task),
                       _sel(tier, kind, drive=drive, task=task))
    return out


def size_comparisons(cells: list[Cell]) -> list[dict]:
    """At every lattice, band mapping, window and channel count, on recognition, the order task and the
    digit-sequence task: the network against each control; on recognition, against the trained
    baselines sized to it, and one band per row against paper 02's 16 bands mapped onto the rows, and
    the longer window against paper 02's."""
    out = _controls(cells, "size", "recognition") + _controls(cells, "size", "order")
    out += _controls(cells, "sequence", "sequence")
    for arch in plan.ANN_ARCHS:
        net = terms.ARMS[f"ann-{arch}"]
        out += compare(cells, f"{NETWORK} minus the {net} sized to it", _sel("size", "field"),
                       lambda c, arch=arch: c.tier == "trained" and c.arm.get("arch") == arch and c.read == READ)
    for kind, name in (("field", NETWORK), ("severed", UNCOUPLED), ("bank", BANK), ("floor", "spectrogram-only "
                                                                                    "baseline")):
        own = _sel("size", kind)
        out += compare(cells, f"{name}: one band per row minus 16 bands mapped onto the rows",
                       lambda c, own=own: own(c) and c.bands == 0 and c.window == 0,
                       lambda c, own=own: own(c) and c.bands == 16, ignore=("bands",))
        out += compare(cells, f"{name}: the longer window minus paper 02's",
                       lambda c, own=own: own(c) and c.window != 0, lambda c, own=own: own(c) and c.window == 0,
                       ignore=("window",))
    return out


def design_comparisons(cells: list[Cell]) -> list[dict]:
    """Each coupling function and geometry minus the reference network, at every size and pathway."""
    out = []
    for tier, ref_tier, drive in (("design", "size", "envelope"), ("design-quadrature", "quadrature", "quadrature"),
                                  ("design-carrier", "carrier", "carrier")):
        for family, shape in plan.designs():
            if (family, shape) == plan.REFERENCE:
                continue
            def a(c, family=family, shape=shape, tier=tier):
                return (c.tier == tier and c.read == READ and c.arm.get("physics") == family
                        and c.arm.get("boundary") == shape)
            label = f"{terms.level(family)}, {shape} minus Kuramoto, torus ({terms.PATHWAYS[drive]} pathway)"
            out += compare(cells, label, a, _sel(ref_tier, "field", drive=drive),
                           extra={"factor": "coupling function and lattice geometry"})
    return out


def pathway_comparisons(cells: list[Cell]) -> list[dict]:
    """On the quadrature and carrier pathways: the network against that pathway's own controls."""
    out = _controls(cells, "quadrature", "recognition", "quadrature", " (quadrature pathway)", bank=False)
    return out + _controls(cells, "carrier", "recognition", "carrier", " (carrier pathway)")


def chance() -> dict:
    """The exact chance levels the report states: recognition 1/10, the order task 1/2, and the
    digit-sequence task's (harness.confirm.protocol.sequence_chance) at each length."""
    return {"recognition": 1 / am.N_CLASSES, "order": 0.5,
            "sequence": {length: pr.sequence_chance(length) for length in pr.SEQUENCE_LENGTHS}}


def summary(cells: list[Cell] | None = None, runs: list[Instruments] | None = None) -> dict:
    if cells is None:
        cells, loaded = load()
        runs = loaded if runs is None else runs
    return {"chance": chance(), "accuracy": accuracies(cells), "instruments": instruments(runs or []),
            "comparisons": size_comparisons(cells) + design_comparisons(cells) + pathway_comparisons(cells)}


# ---------------------------------------------------------------------------
# The readable report: the primary cells
# ---------------------------------------------------------------------------

def _noise(n) -> str:
    return "clean" if n is None else f"{n:+g} dB".replace("+0 dB", "0 dB")


def _acc(r: dict) -> str:
    return f"{r['mean']:.1f}" + (f" ± {r['sd']:.1f}" if r["sd"] is not None else "")


def _diff(r: dict) -> str:
    s = f"{r['mean']:+.2f}" + (f" ± {r['sd']:.2f}" if r["sd"] is not None else "")
    return s + (f" [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]" if "ci95" in r else "")


def _lattice(grid: int, bands: int, window: int) -> str:
    text = terms.lattice_text(grid, bands or None)
    return text + (f", {window:,}-sample window" if window else "")


def _grid_table(records: list[dict], cell) -> list[str]:
    """Rows: lattice, band mapping and window; columns: channel count."""
    rows: dict = defaultdict(dict)
    for r in records:
        rows[(r["grid"], r["bands"], r.get("window", 0))][r["channels"]] = cell(r)
    if not rows:
        return ["(not run yet)"]
    cols = sorted({c for v in rows.values() for c in v})
    out = ["| lattice | " + " | ".join(terms.channels_text(c) for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    return out + [f"| {_lattice(*k)} | " + " | ".join(v.get(c, "") for c in cols) + " |"
                  for k, v in sorted(rows.items())]


def _instrument(name: str):
    def cell(r: dict) -> str:
        v = r.get(name)
        return "" if v is None else f"{v['mean']:.3f}" + (f" ± {v['sd']:.3f}" if v["sd"] is not None else "")
    return cell


def progress() -> dict[str, tuple[int, int, int]]:
    """Runs recorded, runs taken from paper 02, and runs planned, per tier."""
    root, have, out = rn.record_root(), {}, {}
    for name, tier in plan.TIERS.items():
        specs = list(tier())
        for s in specs:
            if s.group() not in have:
                path = root / f"{s.group()}.json"
                have[s.group()] = set(json.loads(path.read_text())["runs"]) if path.exists() else set()
        out[name] = (sum(s.run_id() in have[s.group()] for s in specs),
                     sum(plan.paper02_run(s) is not None for s in specs if plan.reused(s)), len(specs))
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.3g}%"


def _task_section(acc, cmp, projection: str, tier: str, task: str, title: str, note: str, pick=lambda r: True,
                  controls: tuple[str, ...] = ()) -> list[str]:
    """The coupled network's accuracy by lattice and channel count, and its comparisons."""
    read = am.PRIMARY_READ[task]

    def prim(r):
        return (r["width"] == PRIMARY_WIDTH and r["n_train"] == PRIMARY_SIZE and r["projection"] == projection
                and r["task"] == task and pick(r))
    lines = [f"## {title}: coupled oscillator network accuracy ({projection} projection)", "", note, ""]
    lines += _grid_table([r for r in acc if prim(r) and r["tier"] == tier and r["read"] == read
                          and r["arm"].startswith("field-")], _acc) + [""]
    for name in controls:
        lines += [f"### {name} ({title[0].lower() + title[1:]}, {projection} projection)", ""]
        lines += _grid_table([r for r in cmp if prim(r) and r["comparison"] == name], _diff) + [""]
    return lines


def report(s: dict, done: dict[str, tuple[int, int, int]]) -> str:
    acc, cmp, inst, ch = s["accuracy"], s["comparisons"], s["instruments"], s["chance"]
    noise, gain = plan.NOISES[0], plan.GAINS[0]
    lines = ["# Paper 03: results", "",
             "Generated by `uv run python -m harness.confirm.summary`; every number here and more (every width, "
             "pair, position and tier) is in `summary.json`. Accuracies are mean ± standard deviation over three "
             "seeds, in percent. Differences are paired, mean ± standard deviation over seeds, with the 95% "
             "interval from resampling test clips in brackets, in points. Primary cell: width "
             f"{PRIMARY_WIDTH}, {PRIMARY_SIZE:,} training clips, {_noise(noise)}, input gain {gain:g}, the "
             "task's primary read (four windows for recognition, the whole span for the memory tasks). Every "
             "table is given under both projections: the fixed one (paper 02's matrix, the same for every "
             "seed) and the one seeded by each run's seed.", "",
             "Runs recorded: " + "; ".join(f"{k} {d:,} (+{p:,} from paper 02) of {n:,}"
                                           for k, (d, p, n) in done.items())
             + ". A tier not yet complete is summarized over the runs it has.", ""]
    common = [f"{NETWORK} minus the {UNCOUPLED}", f"{NETWORK} minus the {BANK}", f"{NETWORK} minus the {WHOLE}",
              f"{NETWORK} {RATES}"]
    trained = [f"{NETWORK} minus the {terms.ARMS['ann-' + arch]} sized to it" for arch in plan.ANN_ARCHS]
    seq = ch["sequence"]
    for projection in PROJECTIONS:
        lines += _task_section(acc, cmp, projection, "size", "recognition", "Recognition",
                               f"Chance is {_pct(ch['recognition'])}.", controls=tuple(common + trained))
        lines += _task_section(acc, cmp, projection, "size", "order", "Order task, the five pairs pooled",
                               f"Chance is {_pct(ch['order'])}; a read that ignores order stays at chance.",
                               pick=lambda r: r["pair"] == "all", controls=tuple(common))
        for length in pr.SEQUENCE_LENGTHS:
            c = seq[length]
            lines += _task_section(
                acc, cmp, projection, "sequence", "sequence", f"Digit sequences of {length}, mean over positions",
                f"Chance is {_pct(c['per_position'])} at each position; a reader that knows which digits a "
                f"sequence holds but not their order reaches {_pct(c['order_free'])}.",
                pick=lambda r, length=length: r["length"] == length and r["position"] == "mean",
                controls=tuple(common))
            lines += _task_section(
                acc, cmp, projection, "sequence", "sequence", f"Digit sequences of {length}, every position right",
                f"Chance is {_pct(c['whole'])}; knowing the digits but not their order, "
                f"{_pct(c['whole_order_free'])}.",
                pick=lambda r, length=length: r["length"] == length and r["position"] == "all")
        for what in ("one band per row minus 16 bands mapped onto the rows", "the longer window minus paper 02's"):
            name = f"{NETWORK}: {what}"
            lines += [f"## Recognition, {name} ({projection} projection)", ""]
            lines += _grid_table([r for r in cmp if r["comparison"] == name and r["projection"] == projection
                                  and r["width"] == PRIMARY_WIDTH and r["n_train"] == PRIMARY_SIZE], _diff) + [""]
    lines += ["## Synchronization and locking", "",
              "Each network's instruments over its test clips, mean ± standard deviation over seeds: the order "
              "parameter R (1 when every oscillator of a channel shares one phase), each oscillator's phase "
              "locking to its own row's drive (1 when locked), and the share of oscillators whose locking "
              f"exceeds {am.PLV_LOCK_THRESH:g}. Recognition, the "
              "band-energy pathway.", ""]
    for kind, name in (("field-", NETWORK), ("severed-", UNCOUPLED)):
        for key, text in INSTRUMENTS.items():
            if key == "amplitude":
                continue
            lines += [f"### {name}: {text}", ""]
            lines += _grid_table([r for r in inst if r["tier"] == "size" and r["task"] == "recognition"
                                  and r["arm"].startswith(kind)], _instrument(key)) + [""]
    rows = defaultdict(dict)
    for r in inst:
        if r["tier"] in ("design", "size") and r["task"] == "recognition" and r["arm"].startswith("field-") \
                and r["channels"] == am.CHANNELS and not r["bands"] and not r["window"]:
            m = terms._OSCILLATORS.match(terms.split(r["arm"])[0])
            if m and "R" in r:
                rows[(m["physics"], m["boundary"])][r["grid"]] = _instrument("R")(r)
    lines += [f"### Every coupling function and geometry: order parameter R ({terms.channels_text(am.CHANNELS)}, "
              "one band per row)", ""]
    if rows:
        cols = sorted({g for v in rows.values() for g in v})
        lines += ["| coupling function, geometry | " + " | ".join(f"{g} × {g}" for g in cols) + " |",
                  "|---" * (len(cols) + 1) + "|"]
        lines += [f"| {terms.level(fa)}, {sh} | " + " | ".join(v.get(g, "") for g in cols) + " |"
                  for (fa, sh), v in sorted(rows.items())]
    else:
        lines.append("(not run yet)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    s = summary()
    root = rn.record_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    text = report(s, progress())
    (root / "summary.md").write_text(text)
    print(text)
    print(f"wrote {root / 'summary.json'} and {root / 'summary.md'}")


if __name__ == "__main__":
    main()

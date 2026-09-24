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
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from harness.confirm import plan, terms
from harness.confirm import run as rn
from harness.confirm.readout import projection_of, unpack
from harness.measurement.features import PROJECTIONS

PRIMARY_WIDTH, PRIMARY_SIZE = 192, rn.PRIMARY_SIZE
READ = "windowed"
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

    @property
    def grid(self) -> int:
        return self.arm.get("grid", 16)

    @property
    def bands(self) -> int:
        """The band mapping: 0 for one band per row (always 0 at 16 x 16)."""
        b = self.arm.get("bands", 0)
        return 0 if b in (0, self.grid) else b

    @property
    def channels(self) -> int:
        return self.arm.get("channels", 4)

    @property
    def size(self) -> tuple[int, int, int]:
        return self.grid, self.bands, self.channels


def _cells(tier: str, rec: dict, run_id: str, source: str) -> list[Cell]:
    s = rec["spec"]
    label = run_id.split("/")[-1] if s["arm"]["kind"] != "ann" else run_id.split("/")[-2]
    return [Cell(tier, s["drive"], s["noise_db"], s["gain"], s["seed"], s["arm"], label, c["read"], c["width"],
                 c["n_train"], c["acc"], c.get("correct"), rec["n_test"], source, projection_of(c, rec))
            for c in rec["cells"]]


def load() -> list[Cell]:
    """Paper 03's record, and the planned cells paper 02 recorded (unless paper 03 ran them itself)."""
    out, ours = [], set()
    for path in sorted(rn.record_root().glob("*.json")):
        if path.name in NOT_RUNS:
            continue
        for run_id, rec in json.loads(path.read_text())["runs"].items():
            out += _cells(rec["spec"]["tier"], rec, run_id, "paper 03")
            ours.add((path.stem, run_id))
    for tier in plan.TIERS.values():
        for spec in tier():
            mine = (spec.group(), spec.run_id()) in ours
            rec = plan.paper02_run(spec) if plan.reused(spec) and not mine else None
            if rec is not None:
                out += _cells(spec.tier, rec, spec.run_id(), "paper 02")
    return out


def spread(values: dict[str, float]) -> dict:
    """Mean, sample standard deviation and each seed's value, from fractions to points."""
    pts = {k: 100 * v for k, v in sorted(values.items())}
    xs = list(pts.values())
    return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else None,
            "n": len(xs), "values": pts}


def _bits(c: Cell) -> np.ndarray:
    return unpack(c.bits, c.n_test).numpy().astype(np.float64)


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
    if all(a.bits and b.bits for a, b in pairs):
        out["ci95"] = interval(np.mean([_bits(a) - _bits(b) for a, b in pairs], axis=0))
    return out


# ---------------------------------------------------------------------------
# Accuracies
# ---------------------------------------------------------------------------

def accuracies(cells: list[Cell]) -> list[dict]:
    """One record per arm, read, condition, width and size, over its seeds."""
    groups, sources = defaultdict(dict), defaultdict(set)
    for c in cells:
        key = (c.tier, c.drive, c.label, c.read, c.noise, c.gain, c.width, c.n_train, c.projection)
        groups[key][f"seed{c.seed}"] = c.acc
        sources[key].add(c.source)
    fields = ("tier", "drive", "arm", "read", "noise", "gain", "width", "n_train", "projection")
    out = []
    for k, v in sorted(groups.items(), key=str):
        base, channels, grid, bands = terms.split(k[2])
        out.append({**dict(zip(fields, k, strict=True)), "name": terms.arm(k[2], k[0]),
                    "pathway": terms.PATHWAYS[k[1]], "grid": grid, "bands": bands or 0, "channels": channels,
                    "source": sorted(sources[k]), **spread(v)})
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


def compare(cells: list[Cell], name: str, a, b, *, per_channel: bool = True, extra: dict | None = None) -> list[dict]:
    """a minus b over matched cells: `a` and `b` select cells, matched on lattice, band mapping (and
    channel count if `per_channel`), noise, gain, width, training size, seed and projection. A cell
    without gain (the spectrogram-only baseline, a trained baseline) matches the other at every gain,
    and an unprojected cell of b (projection "none") matches a under either projection."""
    bs = [c for c in cells if b(c)]
    b_gain = any(c.gain is not None for c in bs)

    def key(c: Cell, projection: str) -> tuple:
        return (c.drive, c.grid, c.bands, c.channels if per_channel else 0, c.noise,
                c.gain if b_gain else None, c.width, c.n_train, c.seed, projection)
    idx = {key(c, c.projection): c for c in bs}
    groups = defaultdict(list)
    for c in cells:
        if not a(c):
            continue
        other = idx.get(key(c, c.projection)) or idx.get(key(c, "none"))
        if other is not None:
            groups[(c.drive, c.grid, c.bands, c.channels, c.noise, c.gain, c.width, c.n_train,
                    c.projection)].append((c, other))
    return [{"comparison": name, **(extra or {}), "pathway": terms.PATHWAYS[k[0]], "grid": k[1], "bands": k[2],
             "channels": k[3], "noise": k[4], "gain": k[5], "width": k[6], "n_train": k[7], "projection": k[8],
             **paired(v)} for k, v in sorted(groups.items(), key=str)]


def _sel(tier: str, kind: str, read: str = READ, reference: bool | None = True, drive: str | None = None):
    def pick(c: Cell) -> bool:
        return (c.tier == tier and _kind(c) == kind and c.read == read and (drive is None or c.drive == drive)
                and (reference is None or kind not in ("field", "severed") or _is_reference(c) == reference))
    return pick


NETWORK = "coupled oscillator network"


def size_comparisons(cells: list[Cell]) -> list[dict]:
    """At every lattice, band mapping and channel count: the network against each control."""
    whole = _sel("size", "floor", "windowed@wholeclip")
    out = compare(cells, f"{NETWORK} minus the uncoupled oscillator network", _sel("size", "field"),
                  _sel("size", "severed"))
    out += compare(cells, f"{NETWORK} minus the state-matched leaky-integrator bank", _sel("size", "field"),
                   _sel("size", "bank"))
    for kind, name in (("field", NETWORK), ("severed", "uncoupled oscillator network"),
                       ("bank", "state-matched leaky-integrator bank")):
        out += compare(cells, f"{name} minus the spectrogram-only baseline, whole clip", _sel("size", kind),
                       whole, per_channel=False)
    for arch in plan.ANN_ARCHS:
        net = terms.ARMS[f"ann-{arch}"]
        out += compare(cells, f"{NETWORK} minus the {net} sized to it", _sel("size", "field"),
                       lambda c, arch=arch: c.tier == "trained" and c.arm.get("arch") == arch and c.read == READ)
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
    out = []
    for tier in ("quadrature", "carrier"):
        whole = _sel(tier, "floor", "windowed@wholeclip")
        out += compare(cells, f"{NETWORK} minus the spectrogram-only baseline, whole clip ({tier} pathway)",
                       _sel(tier, "field"), whole, per_channel=False)
        out += compare(cells, f"{NETWORK} minus the uncoupled oscillator network ({tier} pathway)",
                       _sel(tier, "field"), _sel(tier, "severed"))
    out += compare(cells, f"{NETWORK} minus the state-matched leaky-integrator bank (carrier pathway)",
                   _sel("carrier", "field"), _sel("carrier", "bank"))
    return out


def summary(cells: list[Cell] | None = None) -> dict:
    cells = load() if cells is None else cells
    return {"accuracy": accuracies(cells),
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


def _lattice(r: dict) -> str:
    return terms.lattice_text(r["grid"], r["bands"] or None)


def _grid_table(records: list[dict], cell) -> list[str]:
    """Rows: lattice and band mapping; columns: channel count."""
    rows: dict = defaultdict(dict)
    for r in records:
        rows[(r["grid"], r["bands"])][r["channels"]] = cell(r)
    if not rows:
        return ["(not run yet)"]
    cols = sorted({c for v in rows.values() for c in v})
    out = ["| lattice | " + " | ".join(terms.channels_text(c) for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    return out + [f"| {terms.lattice_text(g, b or None)} | " + " | ".join(v.get(c, "") for c in cols) + " |"
                  for (g, b), v in sorted(rows.items())]


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


def report(s: dict, done: dict[str, tuple[int, int, int]]) -> str:
    acc, cmp = s["accuracy"], s["comparisons"]

    def prim(r):
        return r["width"] == PRIMARY_WIDTH and r["n_train"] == PRIMARY_SIZE

    lines = ["# Paper 03: results", "",
             "Generated by `uv run python -m harness.confirm.summary`; every number here and more (every width) "
             "is in `summary.json`. Accuracies are mean ± standard deviation over three seeds, in percent. "
             "Differences are paired, mean ± standard deviation over seeds, with the 95% interval from "
             f"resampling test clips in brackets, in points. Primary cell: width {PRIMARY_WIDTH}, "
             f"{PRIMARY_SIZE:,} training clips, the four-window read. Every table is given under both "
             "projections: the fixed one (paper 02's matrix, the same for every seed) and the one seeded by "
             "each run's seed.", "",
             "Runs recorded: " + "; ".join(f"{k} {d:,} (+{p:,} from paper 02) of {n:,}"
                                           for k, (d, p, n) in done.items())
             + ". A tier not yet complete is summarized over the runs it has.", ""]
    names = ["uncoupled oscillator network", "state-matched leaky-integrator bank",
             "spectrogram-only baseline, whole clip"]
    names += [f"{terms.ARMS['ann-' + arch]} sized to it" for arch in plan.ANN_ARCHS]
    for projection in PROJECTIONS:
        for noise in plan.NOISES:
            for gain in plan.GAINS:
                where = f"{_noise(noise)}, gain = {gain:g}, {projection} projection"
                lines += [f"## Size, {where}: coupled oscillator network accuracy", ""]
                lines += _grid_table([r for r in acc if prim(r) and r["tier"] == "size" and r["read"] == READ
                                      and r["noise"] == noise and r["gain"] == gain
                                      and r["projection"] == projection and r["arm"].startswith("field-")],
                                     _acc) + [""]
                for name in names:
                    title = f"{NETWORK} minus the {name}"
                    lines += [f"### {title} ({where})", ""]
                    lines += _grid_table([r for r in cmp if prim(r) and r["comparison"] == title
                                          and r["noise"] == noise and r["gain"] == gain
                                          and r["projection"] == projection], _diff) + [""]
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

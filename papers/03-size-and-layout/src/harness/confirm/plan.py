"""Paper 03's tiers, what they cost, which cells come from paper 02, and the driver that runs them.

    uv run python -m harness.confirm.plan estimate                        # runs, CPU-hours and memory per tier
    uv run python -m harness.confirm.plan prepare                         # the row caches, once
    uv run python -m harness.confirm.plan run size --grids 8 16 32 --dry-run
    uv run python -m harness.confirm.plan run gate size trained --grids 8 16 32 --workers 6

Every tier is a list of specs; nothing here is a free choice at run time.
`--grids` and `--channels` select a stage of a tier (its lattices and channel
counts), so a tier can be run small lattices first; the specs themselves do
not change. The driver skips every spec already recorded, and every spec whose
cell comes from paper 02's record (`reused`), and runs the costliest first so
the pool drains evenly. A spec that fails is logged with its traceback and
the sweep carries on; the failure log is part of the record.

This is a DRAFT plan (REGISTRATION.md is not frozen): the author may drop
gains, pathways, geometries or lattices before it is.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from harness.confirm import protocol as pr
from harness.confirm import run as rn
from harness.confirm.arms import Arm
from harness.utils.paths import PAPER02_ROOT

GRIDS = (8, 16, 32, 64, 128)            # lattices of G x G oscillators per channel
CHANNELS = (1, 2, 4, 8, 16)
#: band mapping: 0 drives each of a lattice's G rows with its own mel band; 16 maps paper 02's 16
#: bands onto the rows. At 16 x 16 the two are the same, and it runs once.
MAPPINGS = (0, 16)
NOISES = (0.0, 5.0)                     # no result is read from clean audio, where the task saturates
GAINS = (1.0, 2.0)
SEEDS = (0, 1, 2)
ANN_ARCHS = ("gru", "tcn", "cnn", "transformer", "s4d")
PHASE_FAMILIES = ("kuramoto", "sakaguchi", "harmonic2", "winfree")
AMPLITUDE_FAMILIES = ("sl", "sl-fixedamp")
SHAPES = ("torus", "cylinder", "sheet", "helix", "cube", "sphere")
CARRIER_GAIN = 32.0                     # the carrier pathway's calibrated gain in paper 02
CARRIER_NOISE = 0.0                     # paper 02 ran the carrier at 0 dB only
#: the carrier integrates 16,000 steps a clip; above 32 x 32 it is not planned (see estimate())
CARRIER_GRIDS = (8, 16, 32)
#: the unprojected (native) read is fitted where the features are small enough, as in paper 02's tier 4
NATIVE_STATES = 1024
REFERENCE = ("kuramoto", "torus")       # paper 02's reference network: its coupling function and geometry

FLOOR_READS = ("windowed", "windowed@wholeclip")
NETWORK_READS = ("windowed",)


def lattices(grids: tuple[int, ...] = GRIDS) -> Iterator[tuple[int, int]]:
    """(grid, bands) for every lattice and band mapping; 16 x 16 once."""
    for grid in grids:
        for bands in MAPPINGS:
            if not (grid == 16 and bands == 16):
                yield grid, bands


def _native(arm: Arm) -> tuple:
    return (rn.PRIMARY_SIZE,) if arm.states <= NATIVE_STATES else ()


def _net(tier: str, drive: str, noise, gain, seed, arm: Arm) -> rn.Spec:
    return rn.Spec(tier, "recognition", drive, noise, gain, seed, arm, native_sizes=_native(arm),
                   reads=NETWORK_READS)


def _floor(tier: str, drive: str, noise, seed, grid: int, bands: int) -> rn.Spec:
    return rn.Spec(tier, "recognition", drive, noise, None, seed, Arm("floor", grid=grid, bands=bands),
                   reads=FLOOR_READS)


def designs(families: tuple[str, ...] = PHASE_FAMILIES + AMPLITUDE_FAMILIES) -> Iterator[tuple[str, str]]:
    """(coupling function, geometry), the Stuart-Landau functions on the torus only, as in paper 02."""
    for family in families:
        for shape in (("torus",) if family in AMPLITUDE_FAMILIES else SHAPES):
            yield family, shape


# ---------------------------------------------------------------------------
# The tiers
# ---------------------------------------------------------------------------

def gate() -> Iterator[rn.Spec]:
    """No input, at every lattice: every read must be exactly chance. Exercises the streamed read."""
    for grid in GRIDS:
        for channels in (1, 16):
            for arm in (Arm("field", channels=channels, grid=grid), Arm("field", channels=channels, grid=grid,
                        severed=True), Arm("bank", channels=channels, grid=grid)):
                yield _net("gate", "envelope", 0.0, 0.0, 0, arm)


def size() -> Iterator[rn.Spec]:
    """The reference network, uncoupled network and state-matched bank at every lattice, channel count
    and band mapping, with the spectrogram-only baseline on the same rows."""
    for grid, bands in lattices():
        for noise in NOISES:
            for seed in SEEDS:
                yield _floor("size", "envelope", noise, seed, grid, bands)
                for channels in CHANNELS:
                    for gain in GAINS:
                        for arm in (Arm("field", channels=channels, grid=grid, bands=bands),
                                    Arm("field", channels=channels, grid=grid, bands=bands, severed=True),
                                    Arm("bank", channels=channels, grid=grid, bands=bands)):
                            yield _net("size", "envelope", noise, gain, seed, arm)


def trained() -> Iterator[rn.Spec]:
    """The five trained baselines, each sized to every network of the size tier, on its rows."""
    for grid, bands in lattices():
        for channels in CHANNELS:
            for arch in ANN_ARCHS:
                for noise in NOISES:
                    for seed in SEEDS:
                        arm = Arm("ann", arch=arch, channels=channels, grid=grid, bands=bands)
                        yield rn.Spec("trained", "recognition", "envelope", noise, None, seed, arm,
                                      reads=("windowed",))


def design() -> Iterator[rn.Spec]:
    """Every other coupling function and geometry at every size, the coupled network only."""
    for grid, bands in lattices():
        for channels in CHANNELS:
            for family, shape in designs():
                if (family, shape) == REFERENCE:
                    continue
                for gain in GAINS:
                    for noise in NOISES:
                        for seed in SEEDS:
                            arm = Arm("field", physics=family, boundary=shape, channels=channels, grid=grid,
                                      bands=bands)
                            yield _net("design", "envelope", noise, gain, seed, arm)


def quadrature() -> Iterator[rn.Spec]:
    """The quadrature pathway at every size: the reference and uncoupled networks, and the pathway's
    own spectrogram-only baseline. The bank has no phase to take it, as in paper 02."""
    for grid, bands in lattices():
        for noise in NOISES:
            for seed in SEEDS:
                yield _floor("quadrature", "quadrature", noise, seed, grid, bands)
                for channels in CHANNELS:
                    for gain in GAINS:
                        for severed in (False, True):
                            arm = Arm("field", channels=channels, grid=grid, bands=bands, severed=severed)
                            yield _net("quadrature", "quadrature", noise, gain, seed, arm)


def carrier() -> Iterator[rn.Spec]:
    """The carrier pathway up to 32 x 32: the reference and uncoupled networks, the state-matched bank at
    the sample rate, and the pathway's own spectrogram-only baseline; 0 dB and gain 32, as in paper 02."""
    for grid, bands in lattices(CARRIER_GRIDS):
        for seed in SEEDS:
            yield _floor("carrier", "carrier", CARRIER_NOISE, seed, grid, bands)
            for channels in CHANNELS:
                for arm in (Arm("field", channels=channels, grid=grid, bands=bands),
                            Arm("field", channels=channels, grid=grid, bands=bands, severed=True),
                            Arm("bank", channels=channels, grid=grid, bands=bands)):
                    yield _net("carrier", "carrier", CARRIER_NOISE, CARRIER_GAIN, seed, arm)


def quadrature_design() -> Iterator[rn.Spec]:
    """The phase coupling functions and geometries on the quadrature pathway, at every size."""
    for grid, bands in lattices():
        for channels in CHANNELS:
            for family, shape in designs(PHASE_FAMILIES):
                if (family, shape) == REFERENCE:
                    continue
                for gain in GAINS:
                    for noise in NOISES:
                        for seed in SEEDS:
                            arm = Arm("field", physics=family, boundary=shape, channels=channels, grid=grid,
                                      bands=bands)
                            yield _net("design-quadrature", "quadrature", noise, gain, seed, arm)


def carrier_design() -> Iterator[rn.Spec]:
    """Every coupling function and geometry on the carrier pathway, up to 32 x 32."""
    for grid, bands in lattices(CARRIER_GRIDS):
        for channels in CHANNELS:
            for family, shape in designs():
                if (family, shape) == REFERENCE:
                    continue
                for seed in SEEDS:
                    arm = Arm("field", physics=family, boundary=shape, channels=channels, grid=grid, bands=bands)
                    yield _net("design-carrier", "carrier", CARRIER_NOISE, CARRIER_GAIN, seed, arm)


TIERS = {"gate": gate, "size": size, "trained": trained, "design": design, "quadrature": quadrature,
         "carrier": carrier, "design-quadrature": quadrature_design, "design-carrier": carrier_design}


# ---------------------------------------------------------------------------
# Cells taken from paper 02's record
# ---------------------------------------------------------------------------
#
# Paper 02 ran the 16 x 16, 4-channel network. Where a paper 03 spec is a run
# paper 02 recorded (same arm, pathway, noise, gain and seed; paper 02's
# primary cell, 2,048 training clips at width 192, is one of its cells), the
# cell is read from paper 02's record rather than run again. Exact kernel
# scaling leaves every 16 x 16 network bit-identical to paper 02's
# (tests/test_kernel_scaling.py), the read of an arm this size is paper 02's
# own code, and `harness.confirm.gates reuse` re-runs a sample and requires
# every accuracy to match.

PAPER02_RECORD = PAPER02_ROOT / "results" / "confirmatory"


TIER1, QUAD02, CARRIER02 = ("tier1-recognition-envelope", "tier3-recognition-quadrature",
                             "tier3-recognition-carrier")


def paper02_group(spec: rn.Spec) -> str | None:
    """The paper 02 record file that holds this spec's run, if paper 02 ran it.

    Paper 02 ran, at 16 x 16 with one band per row: the spectrogram-only
    baseline on every pathway; the reference network, its uncoupled copy and
    both banks (4 and 8 channels, the second being paper 03's state-matched
    bank at 8 channels) on the band-energy pathway; every coupling function
    and geometry at 4 channels (its tier 2); the trained baselines at 2,048
    parameters; and a diagonal of coupling functions and geometries on the
    quadrature and carrier pathways (its tier 3), with the 4-channel bank on
    the carrier.
    """
    a = spec.arm
    if spec.tier == "gate" or a.grid != 16 or a.n_bands != 16:
        return None
    if a.kind == "floor":
        return {"envelope": TIER1, "quadrature": QUAD02, "carrier": CARRIER02}[spec.drive]
    if a.kind == "ann":
        return TIER1 if a.channels == 4 and spec.drive == "envelope" else None
    if a.kind == "bank":
        if spec.drive == "envelope" and a.channels in (4, 8):
            return TIER1
        return CARRIER02 if spec.drive == "carrier" and a.channels == 4 else None
    if a.channels != 4 or (a.omega, a.damping, a.clamp) != ("random", 0.3, 1.0):
        return None
    design = (a.physics, a.boundary)
    if spec.drive == "envelope":
        if design == REFERENCE:
            return TIER1
        return None if a.severed else f"tier2-recognition-envelope-{a.physics}"
    diagonal = not a.severed and (a.boundary == "torus" or design == ("kuramoto", "helix"))
    if spec.drive == "quadrature":
        return QUAD02 if diagonal and a.physics in PHASE_FAMILIES else None
    return CARRIER02 if diagonal else None


def reused(spec: rn.Spec) -> bool:
    return paper02_group(spec) is not None


def paper02_run(spec: rn.Spec) -> dict | None:
    """Paper 02's recorded run for a reused spec, or None if paper 02 has not recorded it (yet)."""
    group = paper02_group(spec)
    if group is None:
        return None
    path = PAPER02_RECORD / f"{group}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["runs"].get(spec.run_id())


# ---------------------------------------------------------------------------
# Cost and memory, for ordering and for the estimate
# ---------------------------------------------------------------------------
#
# Measured on one thread of an Apple M-series CPU (2026-09-23, while other
# work held the remaining cores; tests/ and the scratch probes in the commit
# that added this). Milliseconds per clip for ONE channel of the reference
# network (Kuramoto, torus, band-energy): simulating 61 frames, and computing
# the windowed statistics. Paper 02's tier 2 measured 16 x 16 x 4 at about
# 5 ms a clip, which these reproduce.

SIM_MS = {8: 0.29, 16: 0.97, 32: 2.1, 64: 7.3, 128: 38.5}
FEAT_MS = {8: 0.3, 16: 0.5, 32: 2.4, 64: 7.5, 128: 27.0}
#: relative simulation cost at 64 x 64, one channel
SHAPE_COST = {"torus": 1.0, "cylinder": 1.9, "sheet": 2.4, "helix": 0.74, "cube": 1.66, "sphere": 1.19}
FAMILY_COST = {"kuramoto": 1.0, "sakaguchi": 1.2, "harmonic2": 2.6, "winfree": 2.2, "sl": 1.7,
               "sl-fixedamp": 0.9}
#: the bank simulates about a third as fast as the network, and has half its signals
BANK_SIM, BANK_FEAT = 0.35, 0.5
QUADRATURE_COST = 1.3
CARRIER_STEPS = 16000 / 61              # the carrier integrates at the sample rate
PROJECT_FLOPS = 150e9                   # single-thread matmul throughput measured for the projection
RANDN_PER_S = 48e6                      # single-thread Gaussian draws per second (the projection matrix)
RIDGE_S = 10.0                          # one read's ridge fits at every width, 2,048 clips
#: trained baselines, minutes to train on one thread at 2,048 clips and 30 epochs, measured at three budgets
ANN_MIN = {"gru": (0.6, 0.5, 3.8), "tcn": (0.2, 3.1, 22.7), "cnn": (0.3, 3.4, 22.7),
           "transformer": (0.7, 0.9, 2.0), "s4d": (0.5, 16.2, 154.2)}
ANN_BUDGETS = (2048, 65536, 524288)
CLIPS = rn.PRIMARY_SIZE + 6000


def _ann_minutes(arch: str, budget: int) -> float:
    """Log-log interpolation between the measured budgets (flat below the smallest)."""
    xs, ys = [math.log(b) for b in ANN_BUDGETS], [math.log(m) for m in ANN_MIN[arch]]
    x = math.log(max(budget, ANN_BUDGETS[0]))
    i = 0 if x <= xs[1] else 1
    return math.exp(ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i]))


def seconds(spec: rn.Spec) -> float:
    """Estimated single-thread CPU seconds for one run."""
    a = spec.arm
    if a.kind == "ann":
        return 60 * _ann_minutes(a.arch, a.budget) + RIDGE_S
    if a.kind == "floor":
        return 2.0 + RIDGE_S * len(spec.reads or (1, 1))
    grid = a.grid
    sim, feat = SIM_MS[grid], FEAT_MS[grid]
    if a.kind == "bank":
        sim, feat = sim * BANK_SIM, feat * BANK_FEAT
    else:
        sim *= SHAPE_COST[a.boundary] * FAMILY_COST[a.physics]
    if spec.drive == "quadrature":
        sim *= QUADRATURE_COST
    if spec.drive == "carrier":
        sim, feat = sim * CARRIER_STEPS, feat * CARRIER_STEPS
    per_channel_features = (24 if a.kind == "field" else 12) * grid * grid
    native = per_channel_features * a.channels
    top = min(4096, native)
    project_ms = 1e3 * 2 * per_channel_features * top / PROJECT_FLOPS
    draw_s = native * 4096 / RANDN_PER_S if native > 4096 else 0.0
    return a.channels * CLIPS * (sim + feat + project_ms) / 1e3 + draw_s + RIDGE_S


def memory_gb(spec: rn.Spec) -> float:
    """Rough peak memory of one run: the held features and projection matrix (in memory), or one
    channel's training features and matrix rows (streamed), plus a batch of trajectories."""
    a = spec.arm
    if a.kind in ("floor", "ann"):
        return 1.0
    signals = (2 if a.kind == "field" else 1) * a.grid * a.grid
    frames = 16000 if spec.drive == "carrier" else 61
    if spec.streamed:
        held = (rn.PRIMARY_SIZE + 4096) * 12 * signals * 4
        batch = rn.batch_size(spec, states=a.grid * a.grid) * frames * signals * 4
    else:
        native = 12 * signals * a.channels
        held = CLIPS * native * 4 + native * 4096 * 4 * (native > 4096)
        batch = rn.batch_size(spec) * frames * signals * a.channels * 4
    return (held + batch) / 1e9 + 0.5


# ---------------------------------------------------------------------------
# Preparing and driving
# ---------------------------------------------------------------------------

def _select(specs, grids, channels) -> list[rn.Spec]:
    return [s for s in specs if (not grids or s.arm.grid in grids)
            and (not channels or s.arm.kind in ("floor",) or s.arm.channels in channels)]


def planned(names: list[str], grids=(), channels=()) -> list[rn.Spec]:
    specs = _select([s for name in names for s in TIERS[name]()], grids, channels)
    ids = [(s.group(), s.run_id()) for s in specs]
    if len(set(ids)) != len(ids):
        raise RuntimeError("two specs share a record address; the tier definitions are wrong")
    return specs


def pending(specs: list[rn.Spec]) -> list[rn.Spec]:
    done = {g: rn.recorded_ids(g) for g in {s.group() for s in specs}}
    return sorted((s for s in specs if s.run_id() not in done[s.group()] and not reused(s)),
                  key=seconds, reverse=True)


def estimate(names: list[str] | None = None) -> list[dict]:
    """Per tier and lattice: runs, runs reused from paper 02, CPU-hours on one thread, largest memory."""
    rows = []
    for name in names or list(TIERS):
        by = defaultdict(list)
        for s in TIERS[name]():
            by[s.arm.grid].append(s)
        for grid in sorted(by):
            specs = by[grid]
            rows.append({"tier": name, "grid": grid, "runs": len(specs), "reused": sum(map(reused, specs)),
                         "cpu_hours": sum(seconds(s) for s in specs if not reused(s)) / 3600,
                         "largest_run_hours": max(seconds(s) for s in specs) / 3600,
                         "peak_gb": max(memory_gb(s) for s in specs),
                         "streamed": sum(s.streamed for s in specs)})
    return rows


def _print_estimate(rows: list[dict]) -> None:
    print(f"{'tier':18s} {'grid':>5s} {'runs':>6s} {'reused':>6s} {'streamed':>8s} {'CPU-h':>9s} "
          f"{'largest h':>9s} {'peak GB':>8s}")
    totals = defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        print(f"{r['tier']:18s} {r['grid']:5d} {r['runs']:6d} {r['reused']:6d} {r['streamed']:8d} "
              f"{r['cpu_hours']:9.1f} {r['largest_run_hours']:9.2f} {r['peak_gb']:8.1f}")
        t = totals[r["tier"]]
        t[0] += r["runs"]
        t[1] += r["reused"]
        t[2] += r["cpu_hours"]
    print()
    for tier, (runs, re, hours) in totals.items():
        print(f"{tier:18s} {runs:6d} runs, {re:4d} from paper 02, {hours:9.0f} CPU-hours")
    print(f"{'all':18s} {sum(t[0] for t in totals.values()):6d} runs, "
          f"{sum(t[1] for t in totals.values()):4d} from paper 02, {sum(t[2] for t in totals.values()):9.0f} CPU-hours")


def prepare(workers: int, grids=GRIDS) -> None:
    """Build every row cache the tiers read, in parallel, skipping those that exist."""
    bank = pr.load_bank()
    n = len(bank["labels"])
    counts = sorted({16} | {g for g, b in lattices(grids) if b == 0})
    jobs = [(drive, noise, bands) for drive in pr.CACHED_DRIVES for noise in NOISES for bands in counts
            if pr.load_rows(pr.rows_path(drive, noise, bands), n) is None]
    print(f"=== prepare: {len(jobs)} row cache(s) to build with {workers} worker(s)")
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for f in as_completed([ex.submit(_build, *j) for j in jobs]):
            print(f"    {f.result()}")


def _build(drive, noise, bands) -> str:
    import torch
    torch.set_num_threads(1)
    return str(pr.build_rows(pr.load_bank(), drive, noise, bands).name)


def _work(spec: rn.Spec, device: str, threads: int) -> tuple[str, float]:
    t0 = time.perf_counter()
    return rn.run(spec, device, threads), time.perf_counter() - t0


def drive(names: list[str], workers: int, threads: int, device: str, dry_run: bool,
          grids=(), channels=()) -> None:
    specs = planned(names, grids, channels)
    todo = pending(specs)
    n_reused = sum(map(reused, specs))
    print(f"=== {' + '.join(names)}: {len(specs)} runs planned, {n_reused} from paper 02, "
          f"{len(specs) - len(todo) - n_reused} recorded, {len(todo)} to run on {workers} worker(s) x "
          f"{threads} thread(s), device {device}; ~{sum(map(seconds, todo)) / 3600:.0f} CPU-hours")
    if dry_run:
        for s in todo[:12]:
            print(f"    {s.group()}/{s.run_id()}  ~{seconds(s) / 60:.1f} min, ~{memory_gb(s):.1f} GB"
                  + ("  (streamed)" if s.streamed else ""))
        if len(todo) > 12:
            print(f"    ... and {len(todo) - 12} more")
        return
    failures = rn.record_root() / "failures.log"
    failures.parent.mkdir(parents=True, exist_ok=True)
    t0, done, failed = time.perf_counter(), 0, 0
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        futures = {ex.submit(_work, s, device, threads): s for s in todo}
        for f in as_completed(futures):
            spec = futures[f]
            try:
                where, took = f.result()
                done += 1
            except Exception:                                   # noqa: BLE001 - logged, and the sweep goes on
                failed += 1
                with open(failures, "a") as log:
                    log.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {spec.group()}/{spec.run_id()}\n"
                              f"{traceback.format_exc()}\n")
                where, took = f"FAILED {spec.group()}/{spec.run_id()}", math.nan
            elapsed = time.perf_counter() - t0
            finished = done + failed
            left = elapsed / finished * (len(todo) - finished)
            print(f"[{finished:5d}/{len(todo)}] {elapsed / 60:6.1f} min elapsed, ~{left / 60:6.1f} left"
                  f"  {took:6.1f}s  {where}", flush=True)
    print(f"=== done: {done} recorded, {failed} failed" + (f" (see {failures})" if failed else ""))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="paper 03's tiers")
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("estimate", help="runs, CPU-hours and memory per tier and lattice")
    e.add_argument("tiers", nargs="*", choices=sorted(TIERS))
    p = sub.add_parser("prepare", help="build the row caches the tiers read")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--grids", type=int, nargs="+", default=list(GRIDS))
    r = sub.add_parser("run", help="run one or more tiers")
    r.add_argument("tiers", nargs="+", choices=sorted(TIERS))
    r.add_argument("--grids", type=int, nargs="+", default=[], help="only these lattices (a stage)")
    r.add_argument("--channels", type=int, nargs="+", default=[], help="only these channel counts")
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--device", default="cpu")
    r.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.command == "estimate":
        _print_estimate(estimate(a.tiers or None))
    elif a.command == "prepare":
        prepare(a.workers, tuple(a.grids))
    else:
        drive(a.tiers, a.workers, a.threads, a.device, a.dry_run, tuple(a.grids), tuple(a.channels))


if __name__ == "__main__":
    main()

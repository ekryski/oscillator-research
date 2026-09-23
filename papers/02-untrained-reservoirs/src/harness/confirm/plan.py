"""The registered tiers, and the driver that runs them.

    uv run python -m harness.confirm.plan prepare            # banks' row caches, once
    uv run python -m harness.confirm.plan run tier1 --workers 3 --dry-run
    uv run python -m harness.confirm.plan run gate tier1 --workers 3

Every tier is a list of specs derived from REGISTRATION.md; nothing here is a
free choice at run time. The driver skips every spec already recorded, so a
sweep that stops restarts where it left off, and it runs the costliest specs
first so the pool drains evenly. A spec that fails is logged with its
traceback and the sweep carries on; the failure log is part of the record.
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from harness.confirm import protocol as pr
from harness.confirm import run as rn
from harness.confirm.arms import Arm

NOISES = (None, 0.0, 5.0)
DESIGN_NOISES = (0.0, 5.0)          # no design verdict is read from clean audio, where the task saturates
GAINS = (1.0, 2.0)
SEEDS = (0, 1, 2)
ANN_ARCHS = ("gru", "tcn", "cnn", "transformer", "s4d")
PHASE_FAMILIES = ("kuramoto", "sakaguchi", "harmonic2", "winfree")
AMPLITUDE_FAMILIES = ("sl", "sl-fixedamp")
SHAPES = ("torus", "cylinder", "sheet", "helix", "cube", "sphere")
OMEGAS = ("random", "designed", "uniform")
DAMPINGS = (0.3, 0.1)
CLAMPS = (1.0, 0.5)
CARRIER_GAIN = 32.0                 # the carrier pathway's calibrated gain in the exploratory phase
SIZE_CHANNELS = (1, 4, 16)          # 256, 1,024 and 4,096 states on the same 16 x 16 lattice
BECKER_TRAIN = 18000

FLOOR = Arm("floor")
FIELD = Arm("field")                # Kuramoto, torus, random frequencies, pinning 0.3, clamp 1
SEVERED = Arm("field", severed=True)
BANK_A = Arm("bank", channels=4)    # the field's states and parameters
BANK_B = Arm("bank", channels=8)    # the field's exposed signals, at twice the parameters
FROZEN = (FIELD, SEVERED, BANK_A, BANK_B)


def _ann(arch: str) -> Arm:
    return Arm("ann", arch=arch)


def gate() -> Iterator[rn.Spec]:
    """The g = 0 sanity cells, and the per-clip-span diagnostic."""
    for arm in FROZEN:
        yield rn.Spec("gate", "recognition", "envelope", 0.0, 0.0, 0, arm)
    # how much the exploratory harness's per-clip read window gave the field
    for gain in (0.0, *GAINS):
        for seed in SEEDS:
            yield rn.Spec("gate", "recognition", "envelope", 0.0, gain, seed, FIELD, span="clip")


def tier1() -> Iterator[rn.Spec]:
    """The arms, on recognition at three training sizes and on the order task."""
    for noise in NOISES:
        for seed in SEEDS:
            yield rn.Spec("tier1", "recognition", "envelope", noise, None, seed, FLOOR, sizes=pr.SIZES)
            for gain in GAINS:
                for arm in FROZEN:
                    yield rn.Spec("tier1", "recognition", "envelope", noise, gain, seed, arm, sizes=pr.SIZES)
            for arch in ANN_ARCHS:
                for n in pr.SIZES:
                    yield rn.Spec("tier1", "recognition", "envelope", noise, None, seed, _ann(arch),
                                  sizes=(n,), native_sizes=(n,) if n == rn.PRIMARY_SIZE else ())
    for pair in pr.PAIRS:
        for noise in NOISES:
            for seed in SEEDS:
                yield rn.Spec("tier1", "order", "envelope", noise, None, seed, FLOOR, pair=pair)
                for gain in GAINS:
                    for arm in FROZEN:
                        yield rn.Spec("tier1", "order", "envelope", noise, gain, seed, arm, pair=pair)
                for arch in ANN_ARCHS:
                    yield rn.Spec("tier1", "order", "envelope", noise, None, seed, _ann(arch), pair=pair)


def _design_arms() -> Iterator[Arm]:
    for family in PHASE_FAMILIES + AMPLITUDE_FAMILIES:
        shapes = ("torus",) if family in AMPLITUDE_FAMILIES else SHAPES
        for shape in shapes:
            for omega in OMEGAS:
                for damping in DAMPINGS:
                    for clamp in CLAMPS:
                        yield Arm("field", physics=family, boundary=shape, omega=omega,
                                  damping=damping, clamp=clamp)


def tier2() -> Iterator[rn.Spec]:
    """The design factorial, read at the primary size with and without rotation rates."""
    for noise in DESIGN_NOISES:
        for gain in GAINS:
            for seed in SEEDS:
                for arm in _design_arms():
                    yield rn.Spec("tier2", "recognition", "envelope", noise, gain, seed, arm,
                                  bits="primary", reads=("windowed", "windowed+rate"))


def _diagonal() -> list[Arm]:
    arms = [Arm("field", physics=f, omega=o) for f in PHASE_FAMILIES for o in ("random", "designed")]
    return arms + [Arm("field", boundary="helix", omega=o) for o in ("random", "designed")]


def tier3() -> Iterator[rn.Spec]:
    """The quadrature pathway: a diagonal, since it read near chance throughout the exploratory phase."""
    for noise in DESIGN_NOISES:
        for seed in SEEDS:
            yield rn.Spec("tier3", "recognition", "quadrature", noise, None, seed, FLOOR, bits="primary")
            for gain in GAINS:
                for arm in _diagonal():
                    yield rn.Spec("tier3", "recognition", "quadrature", noise, gain, seed, arm, bits="primary")


def carrier() -> Iterator[rn.Spec]:
    """The carrier pathway's diagonal, with its floor and the leaky bank at the sample rate."""
    arms = _diagonal() + [Arm("field", physics=f, omega=o) for f in AMPLITUDE_FAMILIES
                          for o in ("random", "designed")]
    for seed in SEEDS:
        yield rn.Spec("tier3", "recognition", "carrier", 0.0, None, seed, FLOOR, bits="primary")
        yield rn.Spec("tier3", "recognition", "carrier", 0.0, CARRIER_GAIN, seed, BANK_A, bits="primary")
        for arm in arms:
            yield rn.Spec("tier3", "recognition", "carrier", 0.0, CARRIER_GAIN, seed, arm, bits="primary")


def tier4() -> Iterator[rn.Spec]:
    """Size: the field and its matched leaky bank at 256, 1,024 and 4,096 states."""
    for channels in SIZE_CHANNELS:
        native = (rn.PRIMARY_SIZE,) if channels <= 4 else ()
        for noise in DESIGN_NOISES:
            for seed in SEEDS:
                for arm in (Arm("field", channels=channels), Arm("bank", channels=channels)):
                    yield rn.Spec("tier4", "recognition", "envelope", noise, 2.0, seed, arm,
                                  native_sizes=native, bits="primary")


def becker() -> Iterator[rn.Spec]:
    """Protocol B: Becker et al.'s folds, clean audio, for comparison with published results."""
    for fold in range(len(pr.BECKER_FOLDS)):
        common = dict(protocol="B", fold=fold, sizes=(BECKER_TRAIN,), native_sizes=())
        yield rn.Spec("becker", "recognition", "envelope", None, None, 0, FLOOR, **common)
        for gain in GAINS:
            for arm in FROZEN:
                yield rn.Spec("becker", "recognition", "envelope", None, gain, 0, arm, **common)
        for arch in ANN_ARCHS:
            yield rn.Spec("becker", "recognition", "envelope", None, None, 0, _ann(arch), **common)


TIERS = {"gate": gate, "tier1": tier1, "tier2": tier2, "becker": becker, "tier3": tier3,
         "tier4": tier4, "carrier": carrier}


# ---------------------------------------------------------------------------
# Cost, for ordering
# ---------------------------------------------------------------------------

def cost(spec: rn.Spec) -> float:
    """A rough relative cost, so the pool starts the longest runs first."""
    clips = (pr.ORDER_TRAIN + pr.ORDER_TEST) if spec.task == "order" else max(spec.sizes) + 6000
    per_clip = {"floor": 0.1, "bank": 0.4, "field": 1.0, "ann": 3.0}[spec.arm.kind]
    if spec.drive == "carrier":
        per_clip *= 250
    return clips * per_clip * (spec.arm.channels / 4 if spec.arm.kind != "ann" else 1)


# ---------------------------------------------------------------------------
# Preparing and driving
# ---------------------------------------------------------------------------

def prepare(workers: int) -> None:
    """Build every row cache the tiers read, in parallel, skipping those that exist."""
    bank = pr.load_bank()
    n = len(bank["labels"])
    jobs = [("rows", d, noise, None, None) for d in pr.CACHED_DRIVES for noise in NOISES
            if pr.load_rows(pr.rows_path(d, noise), n) is None]
    for pair in pr.PAIRS:
        for noise in NOISES:
            for code in (0, *(s + 1 for s in SEEDS)):
                size = pr.ORDER_TEST if code == 0 else pr.ORDER_TRAIN
                if pr.load_rows(pr.order_rows_path(pair, code, noise), size) is None:
                    jobs.append(("order", "envelope", noise, pair, code))
    print(f"=== prepare: {len(jobs)} row cache(s) to build with {workers} worker(s)")
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for f in as_completed([ex.submit(_build, *j) for j in jobs]):
            print(f"    {f.result()}")


def _build(kind, drive, noise, pair, code) -> str:
    import torch
    torch.set_num_threads(1)
    bank = pr.load_bank()
    out = (pr.build_rows(bank, drive, noise) if kind == "rows"
           else pr.build_order_rows(bank, pair, code, noise))
    return str(out.name)


def _work(spec: rn.Spec, device: str, threads: int) -> tuple[str, float]:
    t0 = time.perf_counter()
    return rn.run(spec, device, threads), time.perf_counter() - t0


def planned(names: list[str]) -> list[rn.Spec]:
    specs = [s for name in names for s in TIERS[name]()]
    ids = [(s.group(), s.run_id()) for s in specs]
    if len(set(ids)) != len(ids):
        raise RuntimeError("two specs share a record address; the tier definitions are wrong")
    return specs


def pending(specs: list[rn.Spec]) -> list[rn.Spec]:
    done = {g: rn.recorded_ids(g) for g in {s.group() for s in specs}}
    return sorted((s for s in specs if s.run_id() not in done[s.group()]), key=cost, reverse=True)


def drive(names: list[str], workers: int, threads: int, device: str, dry_run: bool) -> None:
    specs = planned(names)
    todo = pending(specs)
    print(f"=== {' + '.join(names)}: {len(specs)} runs planned, {len(specs) - len(todo)} recorded, "
          f"{len(todo)} to run on {workers} worker(s) x {threads} thread(s), device {device}")
    if dry_run:
        for s in todo[:12]:
            print(f"    {s.group()}/{s.run_id()}")
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
    ap = argparse.ArgumentParser(description="the registered confirmatory tiers")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="build the row caches the tiers read")
    p.add_argument("--workers", type=int, default=4)
    r = sub.add_parser("run", help="run one or more tiers")
    r.add_argument("tiers", nargs="+", choices=sorted(TIERS))
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--device", default="cpu")
    r.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.command == "prepare":
        prepare(a.workers)
    else:
        drive(a.tiers, a.workers, a.threads, a.device, a.dry_run)


if __name__ == "__main__":
    main()

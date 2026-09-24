"""The tiers, and the driver that runs them.

    uv run python -m harness.experiment.plan prepare            # banks' row caches, once
    uv run python -m harness.experiment.plan run tier1 --workers 3 --dry-run
    uv run python -m harness.experiment.plan run gate tier1 --workers 3

Every tier is a list of specs derived from DESIGN.md; nothing here is a
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

from harness.experiment import protocol as pr
from harness.experiment import run as rn
from harness.experiment.arms import Arm

NOISES = (None, 0.0, 5.0)
DESIGN_NOISES = (0.0, 5.0)          # clean audio is left out of the design tiers, where the task saturates
GAINS = (1.0, 2.0)
SEEDS = (0, 1, 2)
TRAINED_ARCHS = ("gru", "tcn", "cnn", "transformer", "s4d")
PHASE_COUPLINGS = ("kuramoto", "kuramoto-sakaguchi", "second-harmonic", "winfree")
AMPLITUDE_COUPLINGS = ("stuart-landau", "stuart-landau-fixed")
GEOMETRIES = ("torus", "cylinder", "sheet", "helix", "cube", "sphere")
FREQUENCIES = ("random", "tonotopic", "identical")
RESTORINGS = (0.3, 0.1)
CEILINGS = (1.0, 0.5)
BECKER_TRAIN = 18000
#: the sweep beyond Tier 2's levels: restoring strengths up to the natural frequency, where an
#: oscillator stops rotating; coupling ceilings above 1; and gains up to 12, below the integrator's
#: bound, where drive, natural frequency and coupling together would pass pi radians a step
SWEEP_RESTORINGS = (0.5, 0.8, 1.0)
SWEEP_CEILINGS = (1.5, 2.0)
SWEEP_GAINS = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)

BASELINE = Arm("baseline")
COUPLED = Arm("network")            # Kuramoto, torus, random frequencies, restoring 0.3, ceiling 1
UNCOUPLED = Arm("network", coupled=False)
BANK_STATE = Arm("bank", channels=4)   # the network's states and parameters
BANK_WIDTH = Arm("bank", channels=8)   # the network's exposed signals, at twice the parameters
UNTRAINED = (COUPLED, UNCOUPLED, BANK_STATE, BANK_WIDTH)


def _trained(arch: str) -> Arm:
    return Arm("trained", arch=arch)


def gate() -> Iterator[rn.Spec]:
    """The g = 0 sanity cells, and the per-clip-span diagnostic."""
    for arm in UNTRAINED:
        yield rn.Spec("gate", "recognition", "spectrogram", 0.0, 0.0, 0, arm)
    # how much a per-clip read window gives the network: the leak the fixed window closes
    for gain in (0.0, *GAINS):
        for seed in SEEDS:
            yield rn.Spec("gate", "recognition", "spectrogram", 0.0, gain, seed, COUPLED, span="clip")


def tier1() -> Iterator[rn.Spec]:
    """The arms, on recognition at three training sizes and on the order task."""
    for noise in NOISES:
        for seed in SEEDS:
            yield rn.Spec("tier1", "recognition", "spectrogram", noise, None, seed, BASELINE, sizes=pr.SIZES)
            for gain in GAINS:
                for arm in UNTRAINED:
                    yield rn.Spec("tier1", "recognition", "spectrogram", noise, gain, seed, arm, sizes=pr.SIZES)
            for arch in TRAINED_ARCHS:
                for n in pr.SIZES:
                    yield rn.Spec("tier1", "recognition", "spectrogram", noise, None, seed, _trained(arch),
                                  sizes=(n,), native_sizes=(n,) if n == rn.PRIMARY_SIZE else ())
    for pair in pr.PAIRS:
        for noise in NOISES:
            for seed in SEEDS:
                yield rn.Spec("tier1", "order", "spectrogram", noise, None, seed, BASELINE, pair=pair)
                for gain in GAINS:
                    for arm in UNTRAINED:
                        yield rn.Spec("tier1", "order", "spectrogram", noise, gain, seed, arm, pair=pair)
                for arch in TRAINED_ARCHS:
                    yield rn.Spec("tier1", "order", "spectrogram", noise, None, seed, _trained(arch), pair=pair)


def _design_arms() -> Iterator[Arm]:
    for coupling in PHASE_COUPLINGS + AMPLITUDE_COUPLINGS:
        geometries = ("torus",) if coupling in AMPLITUDE_COUPLINGS else GEOMETRIES
        for geometry in geometries:
            for frequencies in FREQUENCIES:
                for restoring in RESTORINGS:
                    for ceiling in CEILINGS:
                        yield Arm("network", coupling=coupling, geometry=geometry, frequencies=frequencies,
                                  restoring=restoring, ceiling=ceiling)


def tier2() -> Iterator[rn.Spec]:
    """The design factorial, read at the primary size with and without rotation rates."""
    for noise in DESIGN_NOISES:
        for gain in GAINS:
            for seed in SEEDS:
                for arm in _design_arms():
                    yield rn.Spec("tier2", "recognition", "spectrogram", noise, gain, seed, arm,
                                  bits="primary", reads=("windowed", "windowed+rate"))


def _diagonal() -> list[Arm]:
    arms = [Arm("network", coupling=c, frequencies=f) for c in PHASE_COUPLINGS for f in ("random", "tonotopic")]
    return arms + [Arm("network", geometry="helix", frequencies=f) for f in ("random", "tonotopic")]


def tier3() -> Iterator[rn.Spec]:
    """The quadrature pathway: a diagonal of the design rather than the full factorial."""
    for noise in DESIGN_NOISES:
        for seed in SEEDS:
            yield rn.Spec("tier3", "recognition", "quadrature", noise, None, seed, BASELINE, bits="primary")
            for gain in GAINS:
                for arm in _diagonal():
                    yield rn.Spec("tier3", "recognition", "quadrature", noise, gain, seed, arm, bits="primary")


def sweep() -> Iterator[rn.Spec]:
    """Every coupling function at the reference configuration, beyond Tier 2's restoring strengths,
    coupling ceilings and gains. Paired with Tier 2's reference cells: same clips, seeds and reads."""
    common = dict(bits="primary", reads=("windowed", "windowed+rate"))
    for noise in DESIGN_NOISES:
        for seed in SEEDS:
            for coupling in PHASE_COUPLINGS + AMPLITUDE_COUPLINGS:
                for gain in GAINS:
                    for restoring in SWEEP_RESTORINGS:
                        yield rn.Spec("sweep", "recognition", "spectrogram", noise, gain, seed,
                                      Arm("network", coupling=coupling, restoring=restoring), **common)
                    for ceiling in SWEEP_CEILINGS:
                        yield rn.Spec("sweep", "recognition", "spectrogram", noise, gain, seed,
                                      Arm("network", coupling=coupling, ceiling=ceiling), **common)
                for gain in SWEEP_GAINS:
                    yield rn.Spec("sweep", "recognition", "spectrogram", noise, gain, seed,
                                  Arm("network", coupling=coupling), **common)


def cochlea() -> Iterator[rn.Spec]:
    """The coil and the cochlea (harness.models.geometries.coil), for every phase coupling function and
    kind of natural frequencies, at the reference restoring strength and ceiling. Paired with Tier 2's
    torus and helix cells: same clips, seeds and reads."""
    common = dict(bits="primary", reads=("windowed", "windowed+rate"))
    for noise in DESIGN_NOISES:
        for gain in GAINS:
            for seed in SEEDS:
                for geometry in ("coil", "cochlea", "cochlea-matched"):
                    for coupling in PHASE_COUPLINGS:
                        for frequencies in FREQUENCIES:
                            yield rn.Spec("cochlea", "recognition", "spectrogram", noise, gain, seed,
                                          Arm("network", coupling=coupling, geometry=geometry,
                                              frequencies=frequencies), **common)


def projection() -> Iterator[rn.Spec]:
    """Tier 1's reservoir runs again at the primary size, read under the fixed and the seeded projection.

    Added on 2026-09-24 (DESIGN.md decision log): the fixed
    projection is one draw for every seed, so the spread over seeds leaves out
    the projection's own variability. The fixed cells here must equal Tier 1's.
    """
    for noise in NOISES:
        for seed in SEEDS:
            for gain in GAINS:
                for arm in UNTRAINED:
                    yield rn.Spec("projection", "recognition", "spectrogram", noise, gain, seed, arm,
                                  native_sizes=(), bits="primary", reads=("windowed",), projection="both")
    for pair in pr.PAIRS:
        for noise in NOISES:
            for seed in SEEDS:
                for gain in GAINS:
                    for arm in UNTRAINED:
                        yield rn.Spec("projection", "order", "spectrogram", noise, gain, seed, arm, pair=pair,
                                      native_sizes=(), reads=("pooled",), projection="both")


def becker() -> Iterator[rn.Spec]:
    """Protocol B: Becker et al.'s folds, clean audio, for comparison with published results."""
    for fold in range(len(pr.BECKER_FOLDS)):
        common = dict(protocol="B", fold=fold, sizes=(BECKER_TRAIN,), native_sizes=())
        yield rn.Spec("becker", "recognition", "spectrogram", None, None, 0, BASELINE, **common)
        for gain in GAINS:
            for arm in UNTRAINED:
                yield rn.Spec("becker", "recognition", "spectrogram", None, gain, 0, arm, **common)
        for arch in TRAINED_ARCHS:
            yield rn.Spec("becker", "recognition", "spectrogram", None, None, 0, _trained(arch), **common)


TIERS = {"gate": gate, "tier1": tier1, "tier2": tier2, "becker": becker, "tier3": tier3,
         "projection": projection, "sweep": sweep, "cochlea": cochlea}


# ---------------------------------------------------------------------------
# Cost, for ordering
# ---------------------------------------------------------------------------

def cost(spec: rn.Spec) -> float:
    """A rough relative cost, so the pool starts the longest runs first."""
    clips = (pr.ORDER_TRAIN + pr.ORDER_TEST) if spec.task == "order" else max(spec.sizes) + 6000
    per_clip = {"baseline": 0.1, "bank": 0.4, "network": 1.0, "trained": 3.0}[spec.arm.kind]
    return clips * per_clip * (spec.arm.channels / 4 if spec.arm.kind != "trained" else 1)


# ---------------------------------------------------------------------------
# Preparing and driving
# ---------------------------------------------------------------------------

def prepare(workers: int) -> None:
    """Build every row cache the tiers read, in parallel, skipping those that exist."""
    bank = pr.load_bank()
    n = len(bank["labels"])
    jobs = [("rows", p, noise, None, None) for p in pr.CACHED_PATHWAYS for noise in NOISES
            if pr.load_rows(pr.rows_path(p, noise), n) is None]
    for pair in pr.PAIRS:
        for noise in NOISES:
            for code in (0, *(s + 1 for s in SEEDS)):
                size = pr.ORDER_TEST if code == 0 else pr.ORDER_TRAIN
                if pr.load_rows(pr.order_rows_path(pair, code, noise), size) is None:
                    jobs.append(("order", "spectrogram", noise, pair, code))
    print(f"=== prepare: {len(jobs)} row cache(s) to build with {workers} worker(s)")
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for f in as_completed([ex.submit(_build, *j) for j in jobs]):
            print(f"    {f.result()}")


def _build(kind, pathway, noise, pair, code) -> str:
    import torch
    torch.set_num_threads(1)
    bank = pr.load_bank()
    out = (pr.build_rows(bank, pathway, noise) if kind == "rows"
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


def sweep(names: list[str], workers: int, threads: int, device: str, dry_run: bool,
          task: str | None = None) -> None:
    specs = [s for s in planned(names) if task is None or s.task == task]
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
    ap = argparse.ArgumentParser(description="the study's tiers")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="build the row caches the tiers read")
    p.add_argument("--workers", type=int, default=4)
    r = sub.add_parser("run", help="run one or more tiers")
    r.add_argument("tiers", nargs="+", choices=sorted(TIERS))
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--device", default="cpu")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--task", choices=("recognition", "order"),
                   help="run only this task's specs, so two machines can split a tier without sharing a file")
    a = ap.parse_args(argv)
    if a.command == "prepare":
        prepare(a.workers)
    else:
        sweep(a.tiers, a.workers, a.threads, a.device, a.dry_run, a.task)


if __name__ == "__main__":
    main()

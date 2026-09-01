"""Plan and drive a whole experiment sweep.

The grids live here, declaratively, rather than being spelled out in shell. One
place to read what was run, one place for the resume guard, and one place that
knows how to spend the machine it is on:

    uv run python -m harness.sweep envelope        # the primary drive
    uv run python -m harness.sweep all             # everything, in order
    uv run python -m harness.sweep envelope --dry-run
    uv run python -m harness.sweep envelope --workers 4 --threads 2

Resource detection, by default. Runs are independent processes, so the machine
is used by running several at once rather than by threading one harder — past a
few threads a single run stops scaling, while another whole run scales
perfectly. The planner picks a worker count from the core count AND from free
memory, because the carrier drive is sample-rate and its activations are two
orders of magnitude larger than the envelope drive's: on the same laptop, the
right answer is many envelope workers and few carrier ones. Both numbers are
printed before anything starts, and `--workers` overrides.

Progress is live: a single updating line with the completed count, the rate,
and an ETA from the runs that have actually finished, so a multi-hour sweep
tells you where it is. Under a pipe or CI it degrades to one line per run.

Everything is resume-guarded against the results store, so an interrupted sweep
restarts for free and re-running a finished one costs a few seconds of reading
JSON.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from harness import results as store

# canonical physics values: the literature-anchored nonzero inits at which the
# sakaguchi and harmonic2 families are genuinely distinct from kuramoto. Zero is
# the correct TRAINING start but is kuramoto-degenerate when frozen, so an
# untrained matrix run at zero would not be a distinct factor level at all.
ALPHA = 0.7853981633974483  # pi/4
BETA = 0.5
PHASE_FAMILIES = ("kuramoto", "sakaguchi", "harmonic2", "winfree")
SL_FAMILIES = ("sl", "sl-fixedamp")
SHAPES = ("torus", "cylinder", "sheet", "helix", "cube", "sphere")
OMEGAS = ("random", "designed", "uniform")
#: the three operating points every design verdict is read at, as (noise dB, gain)
CONDITIONS = ((0, 2), (5, 2), (5, 1))

#: rough peak resident memory per worker, by drive. The carrier pathway drives
#: the field at 16 kHz rather than at the 62.5 fps hop rate, so its activation
#: tensors are ~250x longer; measured, not guessed.
MEMORY_PER_WORKER_GB = {"envelope": 1.0, "quadrature": 1.2, "carrier": 3.5, "baselines": 1.5}


def physics_flags(family: str) -> list[str]:
    return {
        "kuramoto": ["--core", "phase", "--coupling", "kuramoto"],
        "sakaguchi": ["--core", "phase", "--coupling", "sakaguchi",
                      "--sakaguchi-alpha", str(ALPHA)],
        "harmonic2": ["--core", "phase", "--coupling", "harmonic2",
                      "--harmonic2-beta", str(BETA)],
        "winfree": ["--core", "phase", "--coupling", "winfree"],
        "sl": ["--core", "sl", "--coupling", "kuramoto"],
        "sl-fixedamp": ["--core", "sl-fixedamp", "--coupling", "kuramoto"],
    }[family]


def omega_flags(level: str) -> list[str]:
    return {"random": ["--arms", "frozen"],
            "designed": ["--arms", "designed"],
            "uniform": ["--arms", "frozen", "--omega-uniform"]}[level]


@dataclass
class Planned:
    """One run the sweep intends to execute."""

    kind: str
    flags: list[str]
    label: str
    drive: str = "envelope"
    extras: dict = field(default_factory=dict)

    def argv(self, threads: int) -> list[str]:
        return [sys.executable, "-m", "harness.runner", *self.flags,
                "--kind", self.kind, "--threads", str(threads), "--skip-if-done"]


# ---------------------------------------------------------------------------
# The grids
# ---------------------------------------------------------------------------

def matrix_runs(drive: str, frontend: str) -> Iterator[Planned]:
    """The untrained factorial: physics x geometry x omega x pinning x clamp,
    at all three operating points. The Stuart-Landau cores are torus-only (the
    amplitude channel's boundary handling is a phase-core feature) and the
    quadrature pathway has no Adler hook for them — both documented scopes
    rather than silent gaps."""
    families = PHASE_FAMILIES + (SL_FAMILIES if frontend != "quad" else ())
    for db, gain in CONDITIONS:
        for family in families:
            shapes = ("torus",) if family in SL_FAMILIES else SHAPES
            for shape in shapes:
                for omega in OMEGAS:
                    for lam in ("0.3", "0.1"):
                        for clamp in ("1", "0.5"):
                            yield Planned(
                                "matrix",
                                ["--task", "digits", "--frontend", frontend, "--seeds", "0",
                                 "--probe-windows", "4", "--probe-macro", "4",
                                 *physics_flags(family), *omega_flags(omega),
                                 "--boundary", shape, "--damping", lam, "--clamp", clamp,
                                 "--noise-db", str(db), "--gain", str(gain)],
                                f"{family}-{shape}-{omega}-lam{lam}-clamp{clamp}-{db}db-g{gain}",
                                drive)


def envelope_runs() -> Iterator[Planned]:
    yield from matrix_runs("envelope", "mag")

    # Gain multiplies FRONTEND-RELATIVE units, so it is only meaningful within
    # a pathway. Each point records its per-tick drive-phase increment; a point
    # is integrator-valid only while the max increment stays below pi.
    gate = ["--task", "digits", "--frontend", "mag", "--noise-db", "0",
            "--n-train", "512", "--n-test", "128", "--batch", "8",
            "--probe-windows", "4", "--probe-macro", "4", "--boundary", "torus",
            "--damping", "0.3", "--clamp", "1", "--core", "phase", "--coupling", "kuramoto"]
    for gain in ("1", "2", "4", "8"):
        for omega in ("random", "designed"):
            for seeds in ("0", "0,1,2"):
                yield Planned("gain", [*gate, *omega_flags(omega), "--seeds", seeds,
                                       "--gain", gain],
                              f"gain g{gain} {omega} {'3-seed' if ',' in seeds else '1-seed'}")
    for gain in ("0.5", "1", "2"):  # the amplitude core's own gain response
        yield Planned("gain",
                      ["--task", "digits", "--frontend", "mag", "--noise-db", "5",
                       "--gain", gain, "--seeds", "0", "--core", "sl",
                       "--coupling", "kuramoto", "--boundary", "torus",
                       "--damping", "0.5", "--clamp", "0.5", "--arms", "designed"],
                      f"gain sl g{gain}")

    # the two knobs that set the operating regime: the cap on coupling
    # amplification, and the pinning that pulls each oscillator toward rest
    for clamp in ("0.5", "1", "2", "4", "8"):
        for lam in ("0.05", "0.1", "0.3"):
            yield Planned("clamp-pinning",
                          ["--task", "digits", "--frontend", "mag", "--noise-db", "0",
                           "--gain", "1", "--seeds", "0,1,2", "--probe-windows", "4",
                           "--probe-macro", "4", "--core", "phase", "--coupling", "kuramoto",
                           "--boundary", "torus", "--arms", "frozen",
                           "--clamp", clamp, "--damping", lam],
                          f"clamp {clamp} pinning {lam}")

    # does the translation-invariant stencil matter, or would any connectivity
    # of matched strength do? k = nonzero couplings per oscillator
    dis = ["--task", "digits", "--frontend", "mag", "--noise-db", "0", "--gain", "1",
           "--seeds", "0,1,2", "--probe-windows", "4", "--probe-macro", "4",
           "--boundary", "torus", "--arms", "frozen", "--damping", "0.3", "--clamp", "1"]
    yield Planned("coupling-structure", [*dis, "--core", "phase", "--coupling", "kuramoto"],
                  "coupling circulant")
    for k in ("1", "10", "256"):
        yield Planned("coupling-structure", [*dis, "--core", "randgraph", "--graph-k", k],
                      f"coupling random graph k={k}")

    # alpha = pi/4 and beta = 0.5 are literature-anchored but pointwise; these
    # sweeps say whether the family verdicts survive across the range
    sens = ["--task", "digits", "--frontend", "mag", "--noise-db", "0", "--gain", "2",
            "--seeds", "0", "--probe-windows", "4", "--probe-macro", "4", "--core", "phase",
            "--boundary", "torus", "--arms", "frozen", "--damping", "0.3", "--clamp", "1"]
    for alpha in ("0.3926990816987241", "0.7853981633974483", "1.1780972450961724",
                  "1.39", "1.5707963267948966"):
        yield Planned("sensitivity", [*sens, "--coupling", "sakaguchi",
                                      "--sakaguchi-alpha", alpha],
                      f"sakaguchi alpha={float(alpha):.4f}")
    for beta in ("0.25", "0.5", "1"):
        yield Planned("sensitivity", [*sens, "--coupling", "harmonic2",
                                      "--harmonic2-beta", beta],
                      f"harmonic2 beta={beta}")

    # Two-digit sequences with identical unordered content. The full-span
    # pooled read is order-free by construction, so its floor sits at chance
    # and anything above it must come from state that persists across time.
    ordr = ["--task", "digitpairs", "--frontend", "mag", "--noise-db", "0", "--gain", "1",
            "--seeds", "0,1,2", "--probe-windows", "4", "--probe-macro", "4",
            "--boundary", "torus", "--damping", "0.3"]
    for pair in ("3,7", "1,8", "2,5", "4,9", "0,6"):
        for arm in ("kuramoto", "sl", "gru"):
            core = ["--core", "sl"] if arm == "sl" else ["--core", "phase"]
            arms = ["--arms", "gru"] if arm == "gru" else ["--arms", "frozen"]
            yield Planned("order", [*ordr, "--clamp", "1", *core, "--coupling", "kuramoto",
                                    *arms, "--pair", pair],
                          f"order {pair} {arm}")
    # severed coupling: the spectral cap driven to zero leaves uncoupled
    # oscillators, so whatever memory remains is per-oscillator integration
    for pair in ("3,7", "1,8", "2,5"):
        yield Planned("order", [*ordr, "--clamp", "1e-09", "--core", "phase",
                                "--coupling", "kuramoto", "--arms", "frozen", "--pair", pair],
                      f"order {pair} severed")


def quadrature_runs() -> Iterator[Planned]:
    yield from matrix_runs("quadrature", "quad")
    # if the collapse were a drive-strength problem, pushing the Adler torque
    # far past the locking threshold should recover it
    for gain in ("8", "32", "128"):
        yield Planned("gain",
                      ["--task", "digits", "--frontend", "quad", "--noise-db", "5",
                       "--gain", gain, "--seeds", "0", "--probe-windows", "4",
                       "--core", "phase", "--coupling", "kuramoto", "--boundary", "torus",
                       "--arms", "frozen", "--damping", "0.5", "--clamp", "0.5"],
                      f"gain g{gain}", "quadrature")


def carrier_runs() -> Iterator[Planned]:
    # frontend units differ by ~62x between the mel and carrier paths, so gain
    # equality across pathways is meaningless and each calibrates on its own
    # scale; the sweep also locates the integrator-validity bound
    gate = ["--task", "digits", "--frontend", "carrier", "--noise-db", "0",
            "--n-train", "512", "--n-test", "128", "--batch", "8",
            "--probe-windows", "4", "--probe-macro", "4", "--boundary", "torus",
            "--damping", "0.3", "--clamp", "1", "--core", "phase", "--coupling", "kuramoto"]
    for gain in ("1", "2", "8", "32", "64", "128", "256", "512"):
        for omega in ("random", "designed"):
            yield Planned("gain", [*gate, *omega_flags(omega), "--seeds", "0", "--gain", gain],
                          f"gain g{gain} {omega}", "carrier")
    for gain in ("1", "2", "8", "32", "64"):  # three-seed re-verification, valid points only
        for omega in ("random", "designed"):
            yield Planned("gain", [*gate, *omega_flags(omega), "--seeds", "0,1,2",
                                   "--gain", gain],
                          f"gain g{gain} {omega} 3-seed", "carrier")

    # A diagonal, not a factorial: the envelope and quadrature matrices already
    # bound the geometry and design axes, so this pathway spends its budget on
    # the question it exists to answer — does removing the transcoder restore a
    # dynamical margin? — plus one helix pair to check the geometry verdict.
    diag = ["--task", "digits", "--frontend", "carrier", "--noise-db", "0", "--gain", "32",
            "--seeds", "0", "--batch", "8", "--probe-windows", "4", "--probe-macro", "4",
            "--damping", "0.3", "--clamp", "1"]
    for family in PHASE_FAMILIES + SL_FAMILIES:
        for omega in ("random", "designed"):
            yield Planned("matrix", [*diag, *physics_flags(family), *omega_flags(omega),
                                     "--boundary", "torus"],
                          f"{family}-torus-{omega}", "carrier")
    for omega in ("random", "designed"):
        yield Planned("matrix", [*diag, *physics_flags("kuramoto"), *omega_flags(omega),
                                 "--boundary", "helix"],
                      f"kuramoto-helix-{omega}", "carrier")


def baseline_runs() -> Iterator[Planned]:
    """The reference frame. Every accuracy elsewhere is a difference against
    something measured here, so this runs first."""
    conv = ["--task", "digits", "--frontend", "mag", "--gain", "2.0", "--seeds", "0,1,2",
            "--probe-windows", "4", "--probe-macro", "4", "--select", "best-val",
            "--select-features", "windowed"]
    for db in ("5", "0"):
        for arch in ("gru", "tcn", "cnn", "transformer", "s4d"):
            # s4d needs the learning rate the control below picked
            lr = "3e-4" if arch == "s4d" else "3e-3"
            yield Planned("conventional", [*conv, "--noise-db", db, "--lr", lr, "--arms", arch],
                          f"{arch} at {db} dB", "baselines")

    # short probes reporting optimization health only: does the arm's train
    # loss move at all under the frozen-probe objective?
    lrc = ["--task", "digits", "--frontend", "mag", "--noise-db", "5", "--gain", "2.0",
           "--seeds", "0", "--epochs", "10", "--n-train", "1024", "--n-test", "256"]
    for arch in ("cnn", "tcn", "s4d"):
        # the ReLU feed-forward arms get the extra decade: they are the ones
        # that refuse to move, so the search has to go far enough down to rule
        # out "the learning rate was wrong" as the explanation. S4D was already
        # learning by 3e-4 and monotone in the right direction, so it stops there.
        rates = ("3e-3", "1e-3", "3e-4") if arch == "s4d" else ("3e-3", "1e-3", "3e-4", "1e-4")
        for lr in rates:
            yield Planned("lr-control", [*lrc, "--lr", lr, "--arms", arch],
                          f"{arch} lr={lr}", "baselines")

    # The unmodified samples saturate the models under test, and a near-ceiling
    # task cannot discriminate design points. Pick rule, registered before the
    # grid ran: take the level whose strongest untrained arm lands in [0.70, 0.90].
    for db in ("-10", "-5", "0", "5", "10"):
        for core in ("phase", "sl"):
            yield Planned("noise-calibration",
                          ["--task", "digits", "--frontend", "mag", "--noise-db", db,
                           "--gain", "2.0", "--seeds", "0", "--core", core,
                           "--coupling", "kuramoto", "--boundary", "torus",
                           "--damping", "0.5", "--clamp", "0.5", "--arms", "frozen,designed"],
                          f"{core} at {db} dB", "baselines")

    # same seed in, bit-identical numbers out, or the replication claims
    # everywhere else are worthless
    det = ["--task", "digits", "--frontend", "carrier", "--noise-db", "0", "--gain", "32",
           "--seeds", "0", "--batch", "8", "--n-train", "512", "--n-test", "128",
           "--probe-windows", "4", "--probe-macro", "4", "--boundary", "torus",
           "--core", "phase", "--coupling", "kuramoto", "--arms", "frozen",
           "--damping", "0.3", "--clamp", "1"]
    for rep in ("a", "b"):
        yield Planned("determinism", [*det, "--replicate", rep],
                      f"replicate {rep}", "baselines")


SWEEPS = {
    "baselines": baseline_runs,
    "envelope": envelope_runs,
    "quadrature": quadrature_runs,
    "carrier": carrier_runs,
}


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def total_memory_gb() -> float:
    """Physical RAM, without taking a dependency on psutil for one number."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True, check=True)
            return int(out.stdout.strip()) / 1e9
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1e9
    except (OSError, ValueError, subprocess.SubprocessError):
        return 8.0  # a conservative floor if the platform will not say


def available_memory_gb() -> float:
    """Free-ish memory, so a sweep does not evict everything else on the box."""
    try:
        if platform.system() == "Linux":
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
        if platform.system() == "Darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
            page, free = 4096, 0
            for line in out.stdout.splitlines():
                if "page size of" in line:
                    page = int(line.split("page size of")[1].split()[0])
                if line.startswith(("Pages free:", "Pages inactive:", "Pages speculative:")):
                    free += int(line.split()[-1].rstrip("."))
            return free * page / 1e9
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return total_memory_gb() * 0.5


def plan_resources(sweep: str, workers: int | None, threads: int | None) -> tuple[int, int]:
    """(workers, threads per worker), from the cores and memory actually here.

    Runs are independent, so throughput comes from running several at once
    rather than from threading one harder: a single run stops scaling past a
    few threads, while another whole run scales perfectly. Memory is the other
    wall, and it differs by an order of magnitude between drives.
    """
    cores = os.cpu_count() or 4
    usable = max(1, cores - 1)  # leave one core for the machine's owner
    need = MEMORY_PER_WORKER_GB.get(sweep, 1.5)
    by_memory = max(1, int(available_memory_gb() * 0.75 / need))
    chosen = workers or max(1, min(usable, by_memory))
    per_worker = threads or max(1, usable // chosen)
    return chosen, per_worker


# ---------------------------------------------------------------------------
# Driving
# ---------------------------------------------------------------------------

def human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


class Progress:
    """A live one-line report, or one line per run when output is not a tty."""

    def __init__(self, total: int, tty: bool | None = None):
        self.total, self.done, self.failed = total, 0, 0
        self.start = time.monotonic()
        self.tty = sys.stdout.isatty() if tty is None else tty
        self.width = shutil.get_terminal_size((100, 24)).columns

    def update(self, label: str, ok: bool) -> None:
        self.done += 1
        self.failed += not ok
        elapsed = time.monotonic() - self.start
        rate = self.done / max(elapsed, 1e-9)
        eta = (self.total - self.done) / rate if rate else 0.0
        pct = 100 * self.done / self.total
        flag = "" if ok else "  FAILED"
        line = (f"[{self.done:>5}/{self.total}] {pct:5.1f}%  "
                f"{human(elapsed)} elapsed, {human(eta)} left  "
                f"{rate * 60:.1f}/min  {label}{flag}")
        if self.tty and ok:
            print(f"\r{line[: self.width - 1]:<{self.width - 1}}", end="", flush=True)
        else:
            if self.tty:
                print()
            print(line, flush=True)

    def finish(self) -> None:
        if self.tty:
            print()
        elapsed = time.monotonic() - self.start
        print(f"{self.done} runs in {human(elapsed)}"
              + (f", {self.failed} FAILED" if self.failed else ""))


def execute(planned: list[Planned], workers: int, threads: int, cwd) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    progress = Progress(len(planned))
    failures: list[tuple[str, str]] = []

    def one(run: Planned) -> tuple[Planned, bool, str]:
        proc = subprocess.run(run.argv(threads), cwd=cwd, capture_output=True, text=True)
        return run, proc.returncode == 0, proc.stderr[-2000:]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, r) for r in planned]
        for fut in as_completed(futures):
            run, ok, err = fut.result()
            progress.update(run.label, ok)
            if not ok:
                failures.append((run.label, err))
    progress.finish()
    for label, err in failures[:5]:
        print(f"\n--- {label} ---\n{err}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="plan and drive an experiment sweep")
    ap.add_argument("sweep", choices=("all", *SWEEPS))
    ap.add_argument("--workers", type=int, help="parallel runs (default: from cores and memory)")
    ap.add_argument("--threads", type=int, help="torch threads per run (default: cores / workers)")
    ap.add_argument("--dry-run", action="store_true", help="plan and print, execute nothing")
    ap.add_argument("--no-score", action="store_true", help="skip the scoring pass at the end")
    a = ap.parse_args(argv)

    names = list(SWEEPS) if a.sweep == "all" else [a.sweep]
    src = __import__("pathlib").Path(__file__).resolve().parent.parent
    status = 0
    for name in names:
        planned = list(SWEEPS[name]())
        todo = [r for r in planned if not store.has_run(_config_of(r), r.kind)]
        workers, threads = plan_resources(name, a.workers, a.threads)
        print(f"\n=== {name}: {len(planned)} runs planned, {len(planned) - len(todo)} "
              f"already recorded, {len(todo)} to run")
        if not todo:
            continue
        print(f"    {os.cpu_count()} cores, {available_memory_gb():.0f} GB free -> "
              f"{workers} workers x {threads} threads "
              f"(~{MEMORY_PER_WORKER_GB.get(name, 1.5):.1f} GB each)")
        if a.dry_run:
            for r in todo[:10]:
                print(f"      {r.kind:18s} {r.label}")
            if len(todo) > 10:
                print(f"      ... and {len(todo) - 10} more")
            continue
        status |= execute(todo, workers, threads, cwd=src)

    if not a.dry_run and not a.no_score:
        from harness.measurement import score
        for name in names:
            score.main([name] + (["--figures"] if name == "envelope" else []))
    return status


def _config_of(run: Planned) -> dict:
    """The runner's own parser is the single source of truth for defaults, so
    the resume guard asks it rather than reimplementing the flag semantics."""
    from harness.runner import parse_args
    return vars(parse_args([*run.flags, "--kind", run.kind]))


if __name__ == "__main__":
    raise SystemExit(main())

"""The results record: where a run's numbers go, and how they come back.

One file per (drive variant, coupling law), not one directory per run. A full
factorial is thousands of runs, and a tree of thousands of directories is
unreadable, unbrowsable, and miserable to load. Grouping keeps the whole
envelope kuramoto sweep — every geometry, frequency structure, pinning, clamp,
and operating condition — in a single file you can open and read:

    results/
    ├── baselines/
    │   ├── conventional.json          five architectures, both protocols
    │   ├── lr-control.json            the optimization-health probes
    │   ├── noise-calibration.json     the grid the conditions were picked from
    │   ├── determinism.json           two runs of one configuration
    │   ├── floors.json                the no-dynamics reference lines
    │   └── readout-ladder.json        ridge vs MLP vs transformer by data size
    ├── envelope/
    │   ├── matrix-kuramoto.json       ... one per coupling law
    │   ├── gain.json, order.json, coupling-structure.json,
    │   └── clamp-pinning.json, sensitivity.json
    ├── quadrature/
    └── carrier/

A run's identity comes from the configuration it recorded, never from the order
it happened to run in, so the same experiment always lands in the same place no
matter which script or shard produced it. `tests/test_results.py` asserts that
the derivation and the stored key can never drift apart.

Writes are safe under parallelism. Many worker processes finish runs at once
and land in the same group file, so each write takes an exclusive lock, merges
into the file as it currently is on disk, and swaps the result in atomically.
The cost is a few milliseconds against runs that take tens of seconds.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import statistics
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from harness.utils import paths

DRIVE_OF_FRONTEND = {"mag": "envelope", "quad": "quadrature", "carrier": "carrier"}
DRIVES = ("envelope", "quadrature", "carrier")
SEVERED_CLAMP = 1e-9


def results_root() -> Path:
    """Where the record lives, resolved per call rather than at import.

    `OSC_RESULTS_DIR` is how you reproduce into a fresh tree without touching
    the committed one — and reading it late is what makes that work from a
    test, a shell, or mid-process, instead of only before the first import.
    """
    return Path(os.environ.get("OSC_RESULTS_DIR", paths.RESULTS_DIR))


def artifacts_root() -> Path:
    """Per-run extras (state dumps, instrument tensors): regenerable, large,
    and never part of the record, so they sit outside the committed tree."""
    return results_root() / ".artifacts"


def num(x: float) -> str:
    """Compact fixed-point: 0.5 -> 0.5, 32.0 -> 32, and the severed clamp -> 0."""
    return "0" if x == SEVERED_CLAMP else f"{x:g}"


def physics_of(cfg: dict) -> str:
    """The coupling law, or the core when the core is what varies."""
    core = cfg.get("core", "phase")
    if core == "randgraph":
        return f"randgraph-k{cfg.get('graph_k', 1)}"
    if core in ("sl", "sl-fixedamp"):
        return core
    return cfg.get("coupling", "kuramoto")


def omega_of(cfg: dict) -> str:
    """Which natural-frequency structure this run used."""
    if cfg.get("omega_uniform", False):
        return "uniform"
    return "designed" if "designed" in cfg.get("arms", "") else "random"


def condition_of(cfg: dict) -> str:
    return f"{num(cfg['noise_db'])}db-g{num(cfg['gain'])}"


def address(cfg: dict, kind: str | None = None) -> tuple[str, str]:
    """(group file, run id) for one configuration.

    `kind` names what the run is FOR when the configuration alone cannot say —
    a gate and a matrix point can share every physics value and differ only in
    the question being asked. The sweep grids pass it explicitly; runs recorded
    without one fall back to the matrix.
    """
    drive = DRIVE_OF_FRONTEND[cfg["frontend"]]
    kind = kind or cfg.get("kind") or "matrix"

    if kind in ("conventional", "lr-control", "noise-calibration", "determinism"):
        arch = cfg.get("arms", "?")
        db = num(cfg["noise_db"])
        if kind == "conventional":
            return "baselines/conventional.json", f"{arch}-{db}db-frozenprobe"
        if kind == "lr-control":
            return "baselines/lr-control.json", f"{arch}-lr{cfg['lr']:g}-{db}db"
        if kind == "noise-calibration":
            core = "sl" if cfg.get("core") == "sl" else "phase"
            return "baselines/noise-calibration.json", f"{core}-{db}db"
        return "baselines/determinism.json", cfg.get("replicate", "a")

    seeds = "-3seed" if "," in str(cfg.get("seeds", "0")) else ""
    if kind == "matrix":
        return (f"{drive}/matrix-{physics_of(cfg)}.json",
                f"{cfg['boundary']}-{omega_of(cfg)}-lam{num(cfg['damping'])}"
                f"-clamp{num(cfg['clamp'])}-{condition_of(cfg)}")
    if kind == "gain":
        return (f"{drive}/gain.json",
                f"{physics_of(cfg)}-g{num(cfg['gain'])}-{omega_of(cfg)}"
                f"-{num(cfg['noise_db'])}db{seeds}")
    if kind == "order":
        pair = cfg.get("pair", "3,7").replace(",", "")
        arm = ("severed" if cfg["clamp"] == SEVERED_CLAMP
               else "gru" if cfg.get("arms") == "gru" else physics_of(cfg))
        return f"{drive}/order.json", f"pair{pair}-{arm}"
    if kind == "coupling-structure":
        which = "circulant" if cfg.get("core", "phase") == "phase" else physics_of(cfg)
        return f"{drive}/coupling-structure.json", f"{which}-{condition_of(cfg)}"
    if kind == "clamp-pinning":
        return (f"{drive}/clamp-pinning.json",
                f"clamp{num(cfg['clamp'])}-lam{num(cfg['damping'])}-{condition_of(cfg)}")
    if kind == "sensitivity":
        if cfg.get("coupling") == "sakaguchi":
            return f"{drive}/sensitivity.json", f"sakaguchi-alpha{cfg['sakaguchi_alpha']:.4f}"
        return f"{drive}/sensitivity.json", f"harmonic2-beta{num(cfg['harmonic2_beta'])}"
    raise ValueError(f"unknown run kind '{kind}'")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """One experiment run: where it lives, what it was, and what it measured."""

    group: str          # "envelope/matrix-kuramoto.json"
    run_id: str         # "torus-random-lam0.3-clamp1-0db-g2"
    cfg: dict
    rows: list[dict] = field(default_factory=list)

    @property
    def drive(self) -> str:
        return self.group.split("/")[0]

    @property
    def kind(self) -> str:
        stem = self.group.split("/")[1].removesuffix(".json")
        return stem.split("-")[0] if stem.startswith("matrix") else stem

    @property
    def name(self) -> str:
        return f"{self.group.removesuffix('.json')}/{self.run_id}"

    physics = property(lambda self: physics_of(self.cfg))
    omega = property(lambda self: omega_of(self.cfg))

    @property
    def condition(self) -> tuple[float, float]:
        return self.cfg["noise_db"], self.cfg["gain"]

    def mean(self, key: str, fallback: str | None = None) -> float:
        """Mean over seeds of one accuracy column, in percentage points.

        Runs recorded without `--probe-windows` carry no windowed column, so a
        caller asking for it can name a fallback rather than get a hole.
        """
        vals = [r[key] for r in self.rows if key in r]
        if not vals and fallback:
            vals = [r[fallback] for r in self.rows if fallback in r]
        return 100.0 * statistics.fmean(vals) if vals else math.nan

    def instrument(self, key: str) -> float:
        vals = [r[key] for r in self.rows if key in r]
        return statistics.fmean(vals) if vals else math.nan


def group_path(group: str) -> Path:
    return results_root() / group


def load_group(group: str) -> dict:
    p = group_path(group)
    if not p.exists():
        return {"group": group.removesuffix(".json"), "runs": {}}
    return json.loads(p.read_text())


def iter_runs(drive: str | None = None) -> Iterator[Run]:
    """Every recorded run, optionally restricted to one drive variant."""
    root_dir = results_root()
    roots = [root_dir / drive] if drive else [root_dir / d for d in (*DRIVES, "baselines")]
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*.json")):
            raw = json.loads(p.read_text())
            if "runs" not in raw:
                continue  # a standalone record (floors, the readout ladder)
            group = f"{root.name}/{p.name}"
            for run_id, entry in raw["runs"].items():
                yield Run(group, run_id, entry["config"], entry.get("rows", []))


def has_run(cfg: dict, kind: str | None = None) -> bool:
    """The resume guard: is this exact run already recorded?"""
    group, run_id = address(cfg, kind)
    return run_id in load_group(group).get("runs", {})


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_run(cfg: dict, rows: list[dict], kind: str | None = None) -> str:
    """Merge one run into its group file, safely under parallel writers.

    Read-modify-write under an exclusive lock, then an atomic rename, so a
    crashed or killed worker can never leave a half-written group behind.
    """
    group, run_id = address(cfg, kind)
    path = group_path(group)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".json.lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            data = load_group(group)
            data["drive"] = group.split("/")[0]
            data.setdefault("runs", {})[run_id] = {
                "config": {**cfg, "kind": kind or cfg.get("kind") or "matrix"},
                "rows": rows,
            }
            data["runs"] = dict(sorted(data["runs"].items()))
            tmp = tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False)
            with tmp:
                json.dump(data, tmp, indent=1)
                tmp.write("\n")
            os.replace(tmp.name, path)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return f"{group.removesuffix('.json')}/{run_id}"


def artifact_dir(cfg: dict, kind: str | None = None) -> Path:
    """Where a run's regenerable extras go — never inside the committed tree."""
    group, run_id = address(cfg, kind)
    out = artifacts_root() / group.removesuffix(".json") / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def curve_path(cfg: dict, arm: str, seed: int, kind: str | None = None) -> Path:
    """Training curves are small, few, and worth keeping: one flat directory."""
    group, run_id = address(cfg, kind)
    curves = results_root() / "training-curves"
    curves.mkdir(parents=True, exist_ok=True)
    return curves / f"{group.split('/')[0]}-{run_id}-{arm}-s{seed}.csv"

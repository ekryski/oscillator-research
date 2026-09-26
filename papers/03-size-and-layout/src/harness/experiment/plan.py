"""The experiments, what they cost, which cells come from paper 02, and the driver that runs them.

    uv run python -m harness.experiment.plan estimate                 # runs, CPU- and GPU-hours and memory
    uv run python -m harness.experiment.plan prepare                  # the row caches, once
    uv run python -m harness.experiment.plan run size --grids 8 16 32 --dry-run
    uv run python -m harness.experiment.plan run reuse-check leak-check size trained --grids 8 16 32 --workers 6
    uv run python -m harness.experiment.plan benchmark --device cuda --out ../results/benchmark/cuda.json

Every experiment is a list of specs fixed in advance; nothing here is a free
choice at run time. `--grids`, `--channels` and `--task` select part of an
experiment (its lattices, channel counts or task), so an experiment can be run
small lattices first or split between machines; the specs themselves do not
change. The driver skips every spec already recorded, and every spec whose
cells paper 02's record holds completely (`taken_from_paper02`), and runs the
costliest first so the pool drains evenly. A spec that fails is logged with
its traceback and the sweep carries on; the failure log is part of the record.

The paper's Methods and Appendix B describe the design these experiments
implement; `results/README.md` says which record file holds each.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from collections import OrderedDict, defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from harness.experiment import arms as am
from harness.experiment import protocol as pr
from harness.experiment import readout as ro
from harness.experiment import run as rn
from harness.experiment.arms import Arm
from harness.utils.device import DEVICES, resolve
from harness.utils.paths import PAPER02_ROOT

GRIDS = (8, 16, 32, 64, 128)            # lattices of G x G oscillators per channel
GRID = am.GRID                          # paper 02's lattice, 16 x 16
CHANNELS = (1, 2, 4, 8, 16)
#: band mapping: 0 drives each of a lattice's G rows with its own mel band; 16 maps paper 02's 16
#: bands onto the rows. At 16 x 16 the two are the same, and it runs once.
MAPPINGS = (0, 16)
#: one noise level and one input gain: 0 dB SNR (paper 02 also ran clean audio, where the task nearly
#: saturates, and -5 dB) and gain 1 (paper 02 ran 1 and 2)
NOISES = (0.0,)
GAINS = (1.0,)
SEEDS = (0, 1, 2)
TRAINED_ARCHS = ("gru", "tcn", "cnn", "transformer", "s4d")
PHASE_COUPLINGS = ("kuramoto", "kuramoto-sakaguchi", "second-harmonic", "winfree")
AMPLITUDE_COUPLINGS = ("stuart-landau", "stuart-landau-fixed")
GEOMETRIES = ("torus", "cylinder", "sheet", "helix", "cube", "sphere")
#: paper 02's coil and cochlea, and its cochlea at the coil's average coupling (harness.models.geometries.coil)
COCHLEA_GEOMETRIES = ("coil", "cochlea", "cochlea-matched")
#: the front end's longer real analysis window, as long as the zero-padded transform at that band count;
#: the own-band lattices at 64 and 128 run under both windows in the size, trained and quadrature experiments
LONG_WINDOW = {64: 1024, 128: 2048}
#: the unprojected (native) read is fitted where an arm has this many states or fewer
NATIVE_STATES = 1024
REFERENCE = ("kuramoto", "torus")       # paper 02's reference network: its coupling function and geometry


def lattices(grids: tuple[int, ...] = GRIDS) -> Iterator[tuple[int, int]]:
    """(grid, bands) for every lattice and band mapping; 16 x 16 once."""
    for grid in grids:
        for bands in MAPPINGS:
            if not (grid == 16 and bands == 16):
                yield grid, bands


def front_ends(grids: tuple[int, ...] = GRIDS) -> Iterator[tuple[int, int, int]]:
    """(grid, bands, window): every lattice and band mapping, and the own-band lattices of 64 and 128 again
    under the longer analysis window (0 is paper 02's 512-sample window)."""
    for grid, bands in lattices(grids):
        yield grid, bands, 0
        if bands == 0 and grid in LONG_WINDOW:
            yield grid, bands, LONG_WINDOW[grid]


def _native(arm: Arm) -> tuple:
    return (rn.PRIMARY_SIZE,) if arm.states <= NATIVE_STATES else ()


def reads_for(arm: Arm, task: str) -> tuple[str, ...]:
    """The reads recorded for an arm: the task's primary read; with the rotation rates for a network
    (paper 02's secondary read); from the first frame as well for the spectrogram-only baseline."""
    base = am.PRIMARY_READ[task]
    if arm.kind == "baseline":
        return base, f"{base}@wholeclip"
    return (base, f"{base}+rate") if arm.kind == "network" else (base,)


def _net(experiment: str, pathway: str, noise, gain, seed, arm: Arm, task: str = "recognition", **kw) -> rn.Spec:
    return rn.Spec(experiment, task, pathway, noise, gain, seed, arm, native_sizes=_native(arm),
                   reads=reads_for(arm, task), **kw)


def _baseline(experiment: str, pathway: str, noise, seed, grid: int, bands: int, window: int = 0,
              task: str = "recognition", **kw) -> rn.Spec:
    arm = Arm("baseline", grid=grid, bands=bands, window=window)
    return rn.Spec(experiment, task, pathway, noise, None, seed, arm, reads=reads_for(arm, task), **kw)


def _trained(arch: str, channels: int, grid: int, bands: int, window: int, seed: int,
             experiment: str = "trained") -> rn.Spec:
    arm = Arm("trained", arch=arch, channels=channels, grid=grid, bands=bands, window=window)
    return rn.Spec(experiment, "recognition", "spectrogram", NOISES[0], None, seed, arm,
                   reads=reads_for(arm, "recognition"))


def _reservoirs(channels: int, grid: int, bands: int, window: int = 0, bank: bool = True) -> tuple[Arm, ...]:
    """The coupled and uncoupled networks (and the state-matched bank) of one size."""
    arms = (Arm("network", channels=channels, grid=grid, bands=bands, window=window),
            Arm("network", channels=channels, grid=grid, bands=bands, window=window, coupled=False))
    return arms + ((Arm("bank", channels=channels, grid=grid, bands=bands, window=window),) if bank else ())


def designs(couplings: tuple[str, ...] = PHASE_COUPLINGS + AMPLITUDE_COUPLINGS) -> Iterator[tuple[str, str]]:
    """(coupling function, geometry), the Stuart-Landau functions on the torus only, as in paper 02."""
    for coupling in couplings:
        for geometry in (("torus",) if coupling in AMPLITUDE_COUPLINGS else GEOMETRIES):
            yield coupling, geometry


# ---------------------------------------------------------------------------
# The experiments
# ---------------------------------------------------------------------------

def leak_check() -> Iterator[rn.Spec]:
    """No input (input gain 0), at every lattice, in memory and streamed: with nothing driving it an arm's
    state carries nothing about the clip, so every read should be exactly chance (`summary.leak_check`)."""
    for grid in GRIDS:
        for channels in (1, 16):
            for arm in (Arm("network", channels=channels, grid=grid), Arm("network", channels=channels, grid=grid,
                        coupled=False), Arm("bank", channels=channels, grid=grid)):
                yield _net("leak-check", "spectrogram", 0.0, 0.0, 0, arm)


def size() -> Iterator[rn.Spec]:
    """The reference network, uncoupled network and state-matched bank at every lattice, channel count,
    band mapping and window, with the spectrogram-only baseline on the same rows; and the same arms on
    paper 02's order task at every lattice and band mapping."""
    for grid, bands, window in front_ends():
        for noise in NOISES:
            for seed in SEEDS:
                yield _baseline("size", "spectrogram", noise, seed, grid, bands, window)
                for channels in CHANNELS:
                    for gain in GAINS:
                        for arm in _reservoirs(channels, grid, bands, window):
                            yield _net("size", "spectrogram", noise, gain, seed, arm)
    for grid, bands in lattices():
        for pair in pr.PAIRS:
            for noise in NOISES:
                for seed in SEEDS:
                    yield _baseline("size", "spectrogram", noise, seed, grid, bands, task="order", pair=pair)
                    for channels in CHANNELS:
                        for gain in GAINS:
                            for arm in _reservoirs(channels, grid, bands):
                                yield _net("size", "spectrogram", noise, gain, seed, arm, "order", pair=pair)


def trained() -> Iterator[rn.Spec]:
    """The five trained baselines, each sized to every recognition network of the size experiment, on its rows."""
    for grid, bands, window in front_ends():
        for channels in CHANNELS:
            for arch in TRAINED_ARCHS:
                for seed in SEEDS:
                    yield _trained(arch, channels, grid, bands, window, seed)


def sequence() -> Iterator[rn.Spec]:
    """The digit-sequence task, 2, 3 and 4 digits, for the reference network, uncoupled network and
    state-matched bank at every lattice, channel count and band mapping, with the spectrogram-only
    baseline, which can tell which digits a sequence holds but not where."""
    for grid, bands in lattices():
        for length in pr.SEQUENCE_LENGTHS:
            for noise in NOISES:
                for seed in SEEDS:
                    yield _baseline("sequence", "spectrogram", noise, seed, grid, bands, task="sequence", length=length)
                    for channels in CHANNELS:
                        for gain in GAINS:
                            for arm in _reservoirs(channels, grid, bands):
                                yield _net("sequence", "spectrogram", noise, gain, seed, arm, "sequence",
                                           length=length)


def design() -> Iterator[rn.Spec]:
    """Every other coupling function and geometry at every size, the coupled network only."""
    for grid, bands in lattices():
        for channels in CHANNELS:
            for coupling, geometry in designs():
                if (coupling, geometry) == REFERENCE:
                    continue
                for gain in GAINS:
                    for noise in NOISES:
                        for seed in SEEDS:
                            arm = Arm("network", coupling=coupling, geometry=geometry, channels=channels, grid=grid,
                                      bands=bands)
                            yield _net("design", "spectrogram", noise, gain, seed, arm)


def cochlea() -> Iterator[rn.Spec]:
    """The coil, the cochlea and the cochlea at the coil's average coupling, for every phase coupling
    function, at every lattice, channel count and band mapping, the coupled network only. Paired with the
    design experiment's torus and helix, and with each other, at the same size."""
    for grid, bands in lattices():
        for channels in CHANNELS:
            for geometry in COCHLEA_GEOMETRIES:
                for coupling in PHASE_COUPLINGS:
                    for seed in SEEDS:
                        arm = Arm("network", coupling=coupling, geometry=geometry, channels=channels, grid=grid,
                                  bands=bands)
                        yield _net("cochlea", "spectrogram", NOISES[0], GAINS[0], seed, arm)


def quadrature() -> Iterator[rn.Spec]:
    """The quadrature pathway at every size: the reference and uncoupled networks, and the pathway's
    own spectrogram-only baseline. The bank has no phase to take it, as in paper 02."""
    for grid, bands, window in front_ends():
        for noise in NOISES:
            for seed in SEEDS:
                yield _baseline("quadrature", "quadrature", noise, seed, grid, bands, window)
                for channels in CHANNELS:
                    for gain in GAINS:
                        for arm in _reservoirs(channels, grid, bands, window, bank=False):
                            yield _net("quadrature", "quadrature", noise, gain, seed, arm)


def quadrature_design() -> Iterator[rn.Spec]:
    """The phase coupling functions and geometries on the quadrature pathway, at every size."""
    for grid, bands in lattices():
        for channels in CHANNELS:
            for coupling, geometry in designs(PHASE_COUPLINGS):
                if (coupling, geometry) == REFERENCE:
                    continue
                for gain in GAINS:
                    for noise in NOISES:
                        for seed in SEEDS:
                            arm = Arm("network", coupling=coupling, geometry=geometry, channels=channels, grid=grid,
                                      bands=bands)
                            yield _net("design-quadrature", "quadrature", noise, gain, seed, arm)


def reuse_check() -> Iterator[rn.Spec]:
    """A sample of the runs paper 03 takes from paper 02, run again here, on the CPU: one from each paper 02
    record file and arm kind it takes cells from. Recorded apart (`reuse-check-*.json`), so nothing
    reported comes from it; `summary.reuse_check` compares every cell with paper 02's."""
    ref, recognition = Arm("network"), dict(pathway="spectrogram", noise=NOISES[0], gain=GAINS[0])
    runs = [("recognition", recognition, 0, ref), ("recognition", recognition, 1, Arm("network", coupled=False)),
            ("recognition", recognition, 2, Arm("bank", channels=8)),
            ("recognition", recognition, 0, Arm("network", coupling="winfree", geometry="cube")),
            ("recognition", recognition, 2, Arm("network", coupling="stuart-landau")),
            ("recognition", recognition, 1, Arm("network", coupling="kuramoto-sakaguchi", geometry="sheet")),
            ("recognition", recognition, 1, Arm("network", geometry="cochlea")),
            ("order", recognition, 0, ref),
            ("recognition", {**recognition, "pathway": "quadrature"}, 0, ref)]
    for task, c, seed, arm in runs:
        yield _net("reuse-check", c["pathway"], c["noise"], c["gain"], seed, arm, task,
                   **({"pair": pr.PAIRS[0]} if task == "order" else {}))
    yield _baseline("reuse-check", "spectrogram", NOISES[0], 0, GRID, 0)
    yield _trained("gru", am.CHANNELS, GRID, 0, 0, 1, "reuse-check")


EXPERIMENTS = {"leak-check": leak_check, "reuse-check": reuse_check, "size": size, "trained": trained,
               "sequence": sequence, "design": design, "cochlea": cochlea, "quadrature": quadrature,
               "design-quadrature": quadrature_design}


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
# own code, and the reuse-check experiment re-runs a sample on the CPU, which
# the summary compares cell for cell with paper 02's.
#
# Paper 02's record is read from papers/02-untrained-reservoirs/results/, with
# paper 02's labels, spec keys and run identities (which are paper 03's for
# these runs). OSC_PAPER02_RESULTS points elsewhere, at the results/ of another
# checkout of paper 02.

PAPER02_RECORD = Path(os.environ.get("OSC_PAPER02_RESULTS", PAPER02_ROOT / "results"))
#: paper 02's record files: `<experiment>-<task>`, and its design experiment's one per coupling function
CONTROLS, QUADRATURE02, PROJECTION02 = "controls", "quadrature-recognition", "projection"
#: paper 02's reference configuration's other factors: random natural frequencies, restoring strength 0.3,
#: coupling ceiling 1
REFERENCE_FACTORS = ("random", 0.3, 1.0)


def paper02_group(spec: rn.Spec) -> str | None:
    """The paper 02 record file that holds this spec's run, if paper 02 ran it.

    Paper 02 ran, at 16 x 16 with one band per row, 4 channels, 0 dB and
    gains 1 and 2, three seeds: in its controls experiment, the
    spectrogram-only baseline, the reference network and its uncoupled copy,
    both banks (`bank-state` and `bank-width`, 4 and 8 channels, the second
    being paper 03's state-matched bank at 8 channels) and the trained
    baselines at 2,048 parameters, on recognition, and the same untrained arms
    on the order task; in its design experiment, every coupling function and
    geometry (one file per coupling function); and in its quadrature
    experiment, the pathway's baseline and a diagonal of coupling functions
    and geometries (no uncoupled network); and in its cochlea experiment, the
    coil, the cochlea and the cochlea at the coil's average coupling for the
    four phase coupling functions. It did not run the digit-sequence
    task, or any window but its 512 samples. The leak checks differ in their
    channel counts, so none is shared.
    """
    a = spec.arm
    if spec.experiment == "leak-check" or spec.task == "sequence":
        return None
    if a.grid != GRID or a.n_bands != GRID or a.n_window != am.WINDOW:
        return None
    if spec.task == "order":
        if a.kind == "trained" or (a.kind == "bank" and a.channels not in (4, 8)):
            return None
        if a.kind == "network" and (a.channels != am.CHANNELS or (a.coupling, a.geometry) != REFERENCE
                                    or (a.frequencies, a.restoring, a.ceiling) != REFERENCE_FACTORS):
            return None
        return f"{CONTROLS}-order"
    if a.kind == "baseline":
        return {"spectrogram": f"{CONTROLS}-recognition", "quadrature": QUADRATURE02}[spec.pathway]
    if a.kind == "trained":
        return f"{CONTROLS}-recognition" if a.channels == am.CHANNELS and spec.pathway == "spectrogram" else None
    if a.kind == "bank":
        return f"{CONTROLS}-recognition" if spec.pathway == "spectrogram" and a.channels in (4, 8) else None
    if a.channels != am.CHANNELS or (a.frequencies, a.restoring, a.ceiling) != REFERENCE_FACTORS:
        return None
    design = (a.coupling, a.geometry)
    if spec.pathway == "spectrogram":
        if design == REFERENCE:
            return f"{CONTROLS}-recognition"
        if not a.coupled:
            return None
        if a.geometry in COCHLEA_GEOMETRIES:
            return "cochlea-recognition" if a.coupling in PHASE_COUPLINGS else None
        return f"design-recognition-{a.coupling}"
    diagonal = a.coupled and (a.geometry == "torus" or design == ("kuramoto", "helix"))
    return QUADRATURE02 if diagonal and a.coupling in PHASE_COUPLINGS else None


def reused(spec: rn.Spec) -> bool:
    """Whether the spec's cells are to be taken from paper 02's record. The reuse check's are not: it
    runs them again, to compare."""
    return spec.experiment != "reuse-check" and paper02_group(spec) is not None


#: paper 02's record files read so far, the most recent few (a design file is 6 MB)
_PAPER02_FILES: OrderedDict[tuple, dict] = OrderedDict()
PAPER02_FILES_KEPT = 4


def _paper02_file(group: str) -> dict:
    """One of paper 02's record files, its runs by run id. Every paper 02 experiment has run, so a file
    that is missing means a wrong name, and raises rather than passing for runs paper 02 did not make."""
    path = PAPER02_RECORD / f"{group}.json"
    if not path.exists():
        raise FileNotFoundError(f"no paper 02 record file {path}: the group name is wrong, or "
                                "OSC_PAPER02_RESULTS does not point at paper 02's results/")
    key = (path, path.stat().st_mtime_ns)
    if key in _PAPER02_FILES:
        _PAPER02_FILES.move_to_end(key)
    else:
        while len(_PAPER02_FILES) >= PAPER02_FILES_KEPT:
            _PAPER02_FILES.popitem(last=False)
        _PAPER02_FILES[key] = json.loads(path.read_text())["runs"]
    return _PAPER02_FILES[key]


def _same_run(spec: rn.Spec, recorded: dict) -> bool:
    """Whether paper 02's recorded spec is this spec's run: the same task, pathway, noise, gain, seed and
    arm (paper 02's arms have no size of their own, and are 16 x 16)."""
    arm = spec.arm.as_dict()
    return (recorded["task"], recorded["pathway"], recorded["noise_db"], recorded["gain"], recorded["seed"],
            tuple(recorded.get("pair", ()))) == (spec.task, spec.pathway, spec.noise_db, spec.gain, spec.seed,
                                                 tuple(spec.pair)) and all(
        arm[k] == v for k, v in recorded["arm"].items())


def paper02_run(spec: rn.Spec) -> dict | None:
    """Paper 02's recorded run for a spec paper 02 ran, with every cell tagged by its projection, or
    None if paper 02 did not record it.

    Most of paper 02's experiments recorded only the fixed projection and
    tagged nothing: an untagged cell is read as fixed, or as "none" where it
    was read unprojected. Its projection experiment re-reads its controls
    experiment's reservoirs (recognition and the order task, on the
    spectrogram pathway) under both projections, tagging each cell "fixed",
    "seeded" or "none", under the same run identities; their seeded cells are
    added to the run here.
    """
    group = paper02_group(spec)
    if group is None:
        return None
    rec = _paper02_file(group).get(spec.run_id())
    if rec is None:
        return None
    if not _same_run(spec, rec["spec"]):
        raise ValueError(f"paper 02's {group}/{spec.run_id()} is not the run of {spec}")
    cells = [{**c, "projection": ro.projection_of(c, rec)} for c in rec["cells"]]
    if group.startswith(f"{CONTROLS}-"):
        again = _paper02_file(f"{PROJECTION02}-{spec.task}").get(spec.run_id())
        if again is not None:
            cells += [c for c in again["cells"] if c.get("projection") == "seeded"]
    return {**rec, "cells": cells}


def paper02_complete(spec: rn.Spec, rec: dict | None) -> bool:
    """Whether paper 02's run carries every cell paper 03 reports for the spec: each projected read
    it records (at the primary size, in the reads the spec asks for) under the seeded projection as
    well as the fixed one. Paper 02's cells that are unprojected need no seeded counterpart."""
    if rec is None:
        return False
    wanted = set(spec.reads) if spec.reads else set(am.reads(spec.arm, spec.task))

    def cells(projection: str) -> set:
        return {(c["read"], c["width"]) for c in rec["cells"] if c["projection"] == projection
                and c["n_train"] == rn.PRIMARY_SIZE and c["read"] in wanted}
    return bool(cells("fixed") | cells("none")) and cells("fixed") <= cells("seeded")


def taken_from_paper02(spec: rn.Spec) -> bool:
    """Whether the spec's cells come from paper 02's record: paper 02 ran it, and recorded every cell
    paper 03 reports for it. A spec paper 02 ran but did not record completely (a read without its seeded
    projection) is run here, on the CPU, and that run is the one reported."""
    return reused(spec) and paper02_complete(spec, paper02_run(spec))


# ---------------------------------------------------------------------------
# Cost and memory, for ordering and for the estimate
# ---------------------------------------------------------------------------
#
# Milliseconds per clip for ONE channel of the reference network (Kuramoto,
# torus, spectrogram pathway): simulating 61 frames, and computing the windowed
# statistics, on each device, with the coupling implementation `auto` picks
# there (models/phase.py). Measured on this project's Mac, an M1 Max with 64 GB:
#
#   cpu  one thread, 2026-09-23, while paper 02's runs held the other cores;
#        paper 02's design experiment measured 16 x 16 x 4 at about 5 ms a clip, which
#        these reproduce.
#   mps  its 32-core GPU, 2026-09-24, same load on the CPU, batches of 512
#        clips (8, 16), 256 (32), 64 (64) and 32 (128), after warm-up. The
#        statistics were measured at 16, 64 and 128 and the 8 and 32 values
#        interpolated in proportion to G * G; the bank's likewise.
#
# CUDA was not measured, and the estimate does not cover it.

SIM_MS = {"cpu": {8: 0.29, 16: 0.97, 32: 2.1, 64: 7.3, 128: 38.5},
          "mps": {8: 0.031, 16: 0.03, 32: 0.14, 64: 0.90, 128: 5.5}}
FEAT_MS = {"cpu": {8: 0.3, 16: 0.5, 32: 2.4, 64: 7.5, 128: 27.0},
           "mps": {8: 0.01, 16: 0.034, 32: 0.14, 64: 0.57, 128: 2.05}}
#: relative simulation cost of each geometry and coupling function. CPU: measured at 64 x 64 (FFT). MPS:
#: the dense operator (up to 64 x 64) does not depend on the geometry, and the second harmonic couples
#: four fields instead of two; at 128 x 128 (FFT) the factors measured with FFT at 64 x 64. The coil
#: and the two cochleas were measured on 2026-09-25, on the CPU at 64 x 64 and on MPS with FFT at
#: 128 x 128, against the torus measured with them (their dense operator on MPS, at 64 x 64, costs what
#: the torus's does).
GEOMETRY_COST = {"cpu": {"torus": 1.0, "cylinder": 1.9, "sheet": 2.4, "helix": 0.74, "cube": 1.66, "sphere": 1.19,
                         "coil": 1.47, "cochlea": 1.49, "cochlea-matched": 1.51},
                 "mps-fft": {"torus": 1.0, "cylinder": 1.06, "sheet": 1.66, "helix": 0.21, "cube": 2.43,
                             "sphere": 1.08, "coil": 0.70, "cochlea": 0.75, "cochlea-matched": 0.76}}
COUPLING_COST = {"cpu": {"kuramoto": 1.0, "kuramoto-sakaguchi": 1.2, "second-harmonic": 2.6, "winfree": 2.2},
               "mps": {"kuramoto": 1.0, "kuramoto-sakaguchi": 1.0, "second-harmonic": 2.0, "winfree": 1.0},
               "mps-fft": {"kuramoto": 1.0, "kuramoto-sakaguchi": 0.93, "second-harmonic": 1.08, "winfree": 0.81}}
#: the Stuart-Landau cores: relative to the reference on the CPU (FFT, as in paper 02), and measured
#: outright on MPS (ms per channel and clip; dense up to 64 x 64, FFT at 128; 8 taken from 16, and 32
#: the geometric mean of its neighbours; the fixed-amplitude core assumed the same)
AMPLITUDE_COST = {"cpu": {"stuart-landau": 1.7, "stuart-landau-fixed": 0.9}}
AMPLITUDE_MS_MPS = {8: 0.063, 16: 0.063, 32: 0.31, 64: 1.51, 128: 5.04}
#: the bank, relative to the reference network (CPU), or measured (MPS; 8 and 32 interpolated)
BANK_SIM = {"cpu": 0.35}
BANK_MS_MPS = {8: 0.004, 16: 0.014, 32: 0.05, 64: 0.21, 128: 0.51}
BANK_FEAT = 0.5                         # the bank has one signal per state, the network two
QUADRATURE_COST = 1.3                   # measured on the CPU, assumed on MPS
PROJECT_FLOPS = {"cpu": 150e9, "mps": 1.5e12}   # the projection's matrix products, measured
RANDN_PER_S = 48e6                      # Gaussian draws per second on one CPU thread (the matrices, every device)
TWO_THREADS = 1.65                      # measured speed-up drawing a channel's two matrices in two threads
RIDGE_S = 10.0                          # one read's ridge fits at every width under one projection, CPU float64
#: trained baselines, minutes to train at 2,048 clips and 30 epochs, measured per step at three budgets
TRAINED_MIN = {"cpu": {"gru": (0.6, 0.5, 3.8), "tcn": (0.2, 3.1, 22.7), "cnn": (0.3, 3.4, 22.7),
                   "transformer": (0.7, 0.9, 2.0), "s4d": (0.5, 16.2, 154.2)},
           "mps": {"gru": (0.98, 1.12, 1.09), "tcn": (0.28, 0.81, 0.25), "cnn": (0.17, 0.16, 0.19),
                   "transformer": (0.23, 0.29, 0.46), "s4d": (0.61, 0.97, 6.9)}}
TRAINED_BUDGETS = (2048, 65536, 524288)
CLIPS = rn.PRIMARY_SIZE + 6000
#: frames in a recognition clip, the frames the per-clip times were measured over
FRAMES = 61
COST_DEVICES = ("cpu", "mps")


def _trained_minutes(arch: str, budget: int, device: str) -> float:
    """Log-log interpolation between the measured budgets (flat below the smallest)."""
    xs, ys = [math.log(b) for b in TRAINED_BUDGETS], [math.log(m) for m in TRAINED_MIN[device][arch]]
    x = math.log(max(budget, TRAINED_BUDGETS[0]))
    i = 0 if x <= xs[1] else 1
    return math.exp(ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i]))


def _sim_feat_ms(spec: rn.Spec, device: str) -> tuple[float, float]:
    """Simulation and statistics, ms per channel and clip, for the arm and pathway on a device."""
    a, grid = spec.arm, spec.arm.grid
    sim, feat = SIM_MS[device][grid], FEAT_MS[device][grid]
    if a.kind == "bank":
        sim = sim * BANK_SIM["cpu"] if device == "cpu" else BANK_MS_MPS[grid]
        feat *= BANK_FEAT
    elif a.coupling in ("stuart-landau", "stuart-landau-fixed"):
        sim = sim * AMPLITUDE_COST["cpu"][a.coupling] if device == "cpu" else AMPLITUDE_MS_MPS[grid]
    elif device == "cpu":
        sim *= GEOMETRY_COST["cpu"][a.geometry] * COUPLING_COST["cpu"][a.coupling]
    elif grid > 64:                                      # FFT on MPS
        sim *= GEOMETRY_COST["mps-fft"][a.geometry] * COUPLING_COST["mps-fft"][a.coupling]
    else:
        sim *= COUPLING_COST["mps"][a.coupling]
    if spec.pathway == "quadrature":
        sim *= QUADRATURE_COST
    return sim, feat


def clips_and_frames(spec: rn.Spec) -> tuple[int, int]:
    """(clips a run reads, frames per clip): 8,048 of 61 for recognition, 4,096 of 147 for the order
    task, and 4,096 of 147 to 284 for the digit-sequence task."""
    if spec.task == "order":
        return pr.ORDER_TRAIN + pr.ORDER_TEST, pr.joined_frames(2)
    if spec.task == "sequence":
        return pr.SEQUENCE_TRAIN + pr.SEQUENCE_TEST, pr.joined_frames(spec.length)
    return CLIPS, FRAMES


def features_per_state(arm: Arm, read: str) -> int:
    """Features per state for a read: 3 statistics per signal and window; a network has two signals per
    oscillator, and its rotation rates add two more per oscillator and window."""
    windows = am.WINDOWS["recognition"] if read.startswith("windowed") else 1
    per = 3 * windows * (2 if arm.kind == "network" else 1)
    return per + (2 * windows if read.endswith("+rate") else 0)


def seconds(spec: rn.Spec, device: str = "cpu", trained_device: str = "cpu") -> float:
    """Estimated seconds for one run: on one CPU thread (`cpu`) or on the M1 Max's GPU (`mps`).

    A trained baseline trains on `trained_device`, the CPU unless a run asks
    otherwise (`plan run --trained-device`), and a run paper 02 made runs on the
    CPU whatever the device (`_work`). Both include what runs on the CPU either
    way: drawing the two projection matrices, and the ridge readout under each
    projection, for every read (and, on the digit-sequence task, every
    position). The frame counts and the number of clips scale the per-clip times
    measured on recognition; a streamed arm simulates its channels once per read.
    """
    if paper02_group(spec) is not None:
        device = trained_device = "cpu"
    a = spec.arm
    reads = spec.reads or tuple(am.reads(a, spec.task))
    positions = spec.length if spec.task == "sequence" else 1
    ridge = RIDGE_S * len(reads) * 2 * positions             # the fixed and the seeded projection
    clips, frames = clips_and_frames(spec)
    scale = frames / FRAMES
    if a.kind == "trained":
        where = "mps" if trained_device == "mps" else "cpu"
        return 60 * _trained_minutes(a.arch, a.budget, where) * scale * clips / CLIPS + ridge
    if a.kind == "baseline":
        return 2.0 + ridge
    sim, feat = _sim_feat_ms(spec, device)
    sites = a.grid * a.grid
    passes = len(reads) if spec.streamed else 1
    project_ms = draws = 0.0
    for read in reads:
        per_channel = features_per_state(a, read) * sites
        native = per_channel * a.channels
        project_ms += 2 * 1e3 * 2 * per_channel * min(4096, native) / PROJECT_FLOPS[device]
        draws += 2 * native * 4096 / RANDN_PER_S if native > 4096 else 0.0
    if spec.streamed:
        draws /= TWO_THREADS                                  # a channel's two matrices are drawn in two threads
    per_clip_ms = passes * sim * scale + feat * scale * (1 + 0.33 * (len(reads) > 1)) + project_ms
    return a.channels * clips * per_clip_ms / 1e3 + draws + ridge


def memory_gb(spec: rn.Spec) -> float:
    """Rough peak memory of one run: the held features and the two projection matrices (in memory), or
    one channel's training features and its rows of both matrices for the widest read (streamed), plus
    a batch of trajectories. On MPS this is the unified memory the CPU and GPU share."""
    a = spec.arm
    if a.kind in ("baseline", "trained"):
        return 1.0
    signals = (2 if a.kind == "network" else 1) * a.grid * a.grid
    clips, frames = clips_and_frames(spec)
    reads = spec.reads or tuple(am.reads(a, spec.task))
    widest = max(features_per_state(a, r) for r in reads) * a.grid * a.grid
    if spec.streamed:
        held = (rn.PRIMARY_SIZE + 2 * 4096) * widest * 4
        batch = rn.batch_size(spec, states=a.grid * a.grid) * frames * signals * 4
    else:
        native = sum(features_per_state(a, r) for r in reads) * a.grid * a.grid * a.channels
        held = clips * native * 4 + 2 * widest * a.channels * 4096 * 4 * (widest * a.channels > 4096)
        batch = rn.batch_size(spec) * frames * signals * a.channels * 4
    return (held + batch) / 1e9 + 0.5


# ---------------------------------------------------------------------------
# Preparing and driving
# ---------------------------------------------------------------------------

def _select(specs, grids, channels) -> list[rn.Spec]:
    return [s for s in specs if (not grids or s.arm.grid in grids)
            and (not channels or s.arm.kind == "baseline" or s.arm.channels in channels)]


def planned(names: list[str], grids=(), channels=(), task: str | None = None) -> list[rn.Spec]:
    specs = _select([s for name in names for s in EXPERIMENTS[name]()], grids, channels)
    specs = [s for s in specs if task is None or s.task == task]
    ids = [(s.group(), s.run_id()) for s in specs]
    if len(set(ids)) != len(ids):
        raise RuntimeError("two specs share a record address; the experiment definitions are wrong")
    return specs


def pending(specs: list[rn.Spec]) -> list[rn.Spec]:
    """Specs not yet recorded here, and not taken from paper 02's record. A spec paper 02 ran but did
    not record completely is run here instead, and that run is the one reported."""
    done = {g: rn.recorded_ids(g) for g in {s.group() for s in specs}}
    return sorted((s for s in specs if s.run_id() not in done[s.group()] and not taken_from_paper02(s)),
                  key=seconds, reverse=True)


def estimate(names: list[str] | None = None) -> list[dict]:
    """Per experiment and lattice: runs, runs taken from paper 02, hours on one CPU thread and on the M1
    Max's GPU, the longest run on each, and the largest memory."""
    rows = []
    for name in names or list(EXPERIMENTS):
        by = defaultdict(list)
        for s in EXPERIMENTS[name]():
            by[s.arm.grid].append(s)
        for grid in sorted(by):
            specs = by[grid]
            todo = [s for s in specs if not taken_from_paper02(s)]
            row = {"experiment": name, "grid": grid, "runs": len(specs), "reused": len(specs) - len(todo),
                   "peak_gb": max(memory_gb(s) for s in specs), "streamed": sum(s.streamed for s in specs)}
            for device in COST_DEVICES:
                row[f"{device}_hours"] = sum(seconds(s, device) for s in todo) / 3600
                row[f"{device}_longest_hours"] = max(seconds(s, device) for s in specs) / 3600
            rows.append(row)
    return rows


def _print_estimate(rows: list[dict]) -> None:
    print(f"{'experiment':18s} {'grid':>5s} {'runs':>6s} {'reused':>6s} {'streamed':>8s} {'CPU-h':>9s} "
          f"{'MPS-h':>8s} {'longest CPU':>11s} {'longest MPS':>11s} {'peak GB':>8s}")
    totals = defaultdict(lambda: [0, 0, 0.0, 0.0])
    for r in rows:
        print(f"{r['experiment']:18s} {r['grid']:5d} {r['runs']:6d} {r['reused']:6d} {r['streamed']:8d} "
              f"{r['cpu_hours']:9.1f} {r['mps_hours']:8.1f} {r['cpu_longest_hours']:10.2f}h "
              f"{r['mps_longest_hours']:10.2f}h {r['peak_gb']:8.1f}")
        t = totals[r["experiment"]]
        t[0] += r["runs"]
        t[1] += r["reused"]
        t[2] += r["cpu_hours"]
        t[3] += r["mps_hours"]
    print()
    for name, (runs, re, cpu, mps) in totals.items():
        print(f"{name:18s} {runs:6d} runs, {re:4d} from paper 02, {cpu:8.0f} CPU-hours, {mps:7.1f} MPS-hours")
    print(f"{'all':18s} {sum(t[0] for t in totals.values()):6d} runs, {sum(t[1] for t in totals.values()):4d} "
          f"from paper 02, {sum(t[2] for t in totals.values()):8.0f} CPU-hours, "
          f"{sum(t[3] for t in totals.values()):7.1f} MPS-hours")


def cache_jobs(grids=GRIDS) -> list[tuple]:
    """Every row cache the experiments read: the bank's rows on the spectrogram and quadrature pathways at each
    band count and window, and each order-task and digit-sequence set (its test set and every seed's
    training set) at each band count."""
    counts = sorted({16} | {g for g, b in lattices(grids) if b == 0})
    windows = {(g, w) for g, b, w in front_ends(grids) if w}
    codes = (0, *(seed + 1 for seed in SEEDS))
    jobs = [("rows", pathway, noise, bands, pr.HOP_N_FFT) for pathway in pr.CACHED_PATHWAYS for noise in NOISES
            for bands in counts]
    jobs += [("rows", pathway, noise, g, w) for pathway in pr.CACHED_PATHWAYS for noise in NOISES
             for g, w in sorted(windows)]
    jobs += [("order", pair, code, noise, bands) for pair in pr.PAIRS for code in codes for noise in NOISES
             for bands in counts]
    jobs += [("sequence", length, code, noise, bands) for length in pr.SEQUENCE_LENGTHS for code in codes
             for noise in NOISES for bands in counts]
    return jobs


def cache_path(job: tuple):
    kind, *args = job
    if kind == "rows":
        return pr.rows_path(*args)
    return (pr.order_rows_path if kind == "order" else pr.sequence_rows_path)(*args)


def cache_clips(job: tuple, bank_clips: int) -> int:
    if job[0] == "rows":
        return bank_clips
    code = job[2]
    return (pr.ORDER_TEST if code == 0 else pr.ORDER_TRAIN) if job[0] == "order" else (
        pr.SEQUENCE_TEST if code == 0 else pr.SEQUENCE_TRAIN)


def prepare(workers: int, grids=GRIDS) -> None:
    """Build every row cache the experiments read, in parallel, skipping those that exist."""
    n = len(pr.load_bank()["labels"])
    jobs = [j for j in cache_jobs(grids) if pr.load_rows(cache_path(j), cache_clips(j, n)) is None]
    print(f"=== prepare: {len(jobs)} row cache(s) to build with {workers} worker(s)")
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for f in as_completed([ex.submit(_build, j) for j in jobs]):
            print(f"    {f.result()}")


def _build(job: tuple) -> str:
    import torch
    torch.set_num_threads(1)
    kind, *args = job
    build = {"rows": pr.build_rows, "order": pr.build_order_rows, "sequence": pr.build_sequence_rows}[kind]
    return str(build(pr.load_bank(), *args).name)


def _work(spec: rn.Spec, device: str, threads: int, trained_device: str = "cpu") -> tuple[str, float]:
    """One run. A run paper 02 made (the reuse check's, or one paper 02 did not record completely) runs
    on the CPU, the only device on which it is bit-identical to paper 02's."""
    t0 = time.perf_counter()
    if paper02_group(spec) is not None:
        device = trained_device = "cpu"
    return rn.run(spec, device, threads, trained_device), time.perf_counter() - t0


def drive(names: list[str], workers: int, threads: int, device: str, dry_run: bool,
          grids=(), channels=(), trained_device: str = "cpu", task: str | None = None) -> None:
    device, trained_device = resolve(device), resolve(trained_device)
    specs = planned(names, grids, channels, task)
    todo = pending(specs)
    n_reused = sum(map(taken_from_paper02, specs))
    cost = "mps" if device == "mps" else "cpu"
    hours = sum(seconds(s, cost, trained_device) for s in todo) / 3600
    print(f"=== {' + '.join(names)}: {len(specs)} runs planned, {n_reused} from paper 02, "
          f"{len(specs) - len(todo) - n_reused} recorded, {len(todo)} to run on {workers} worker(s) x "
          f"{threads} thread(s), device {device}; ~{hours:.0f} {cost.upper()}-hours")
    if dry_run:
        for s in todo[:12]:
            print(f"    {s.group()}/{s.run_id()}  ~{seconds(s, cost, trained_device) / 60:.1f} min, "
                  f"~{memory_gb(s):.1f} GB"
                  + ("  (streamed)" if s.streamed else ""))
        if len(todo) > 12:
            print(f"    ... and {len(todo) - 12} more")
        return
    failures = rn.record_root() / "failures.log"
    failures.parent.mkdir(parents=True, exist_ok=True)
    t0, done, failed = time.perf_counter(), 0, 0
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        futures = {ex.submit(_work, s, device, threads, trained_device): s for s in todo}
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
    ap = argparse.ArgumentParser(description="paper 03's experiments")
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("estimate", help="runs, CPU- and GPU-hours and memory per experiment and lattice")
    e.add_argument("experiments", nargs="*", choices=sorted(EXPERIMENTS))
    p = sub.add_parser("prepare", help="build the row caches the experiments read")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--grids", type=int, nargs="+", default=list(GRIDS))
    r = sub.add_parser("run", help="run one or more experiments")
    r.add_argument("experiments", nargs="+", choices=sorted(EXPERIMENTS))
    r.add_argument("--grids", type=int, nargs="+", default=[], help="only these lattices (a stage)")
    r.add_argument("--channels", type=int, nargs="+", default=[], help="only these channel counts")
    r.add_argument("--task", choices=("recognition", "order", "sequence"),
                   help="only this task's specs, so two machines can split an experiment without sharing a file")
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--device", default="auto", choices=DEVICES,
                   help="auto: CUDA if present, else Apple Silicon's GPU (mps), else the CPU; only cpu is "
                        "bit-identical to paper 02")
    r.add_argument("--trained-device", default="cpu", choices=DEVICES,
                   help="where the trained baselines train: the CPU (the default, as in paper 02) unless the "
                        "benchmark shows a GPU faster for them")
    r.add_argument("--dry-run", action="store_true")
    b = sub.add_parser("benchmark", help="time one batch of each network on a device, and extrapolate every run")
    b.add_argument("--device", default="auto", choices=DEVICES)
    b.add_argument("--grids", type=int, nargs="+", default=list(GRIDS))
    b.add_argument("--channels", type=int, nargs="+", default=list(CHANNELS))
    b.add_argument("--pathways", nargs="+", default=["spectrogram", "quadrature"],
                   choices=["spectrogram", "quadrature"])
    b.add_argument("--no-designs", action="store_true", help="skip timing the other coupling functions and geometries")
    b.add_argument("--no-trained", action="store_true", help="skip timing the trained baselines' training steps")
    b.add_argument("--max-clips", type=int, default=512, help="clips per timed batch at most")
    b.add_argument("--out", type=Path, help="the JSON report")
    a = ap.parse_args(argv)
    if a.command == "estimate":
        _print_estimate(estimate(a.experiments or None))
    elif a.command == "prepare":
        prepare(a.workers, tuple(a.grids))
    elif a.command == "benchmark":
        from harness.experiment.benchmark import benchmark
        benchmark(a.device, tuple(a.grids), tuple(a.channels), tuple(a.pathways), not a.no_designs, a.max_clips,
                  a.out, trained=not a.no_trained)
    else:
        drive(a.experiments, a.workers, a.threads, a.device, a.dry_run, tuple(a.grids), tuple(a.channels),
              a.trained_device, a.task)


if __name__ == "__main__":
    main()

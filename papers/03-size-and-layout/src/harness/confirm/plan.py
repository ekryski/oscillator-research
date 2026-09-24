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

DESIGN.md states the design these tiers implement, and its open questions.
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

from harness.confirm import arms as am
from harness.confirm import protocol as pr
from harness.confirm import readout as ro
from harness.confirm import run as rn
from harness.confirm.arms import Arm
from harness.utils.device import DEVICES, resolve
from harness.utils.paths import PAPER02_ROOT

GRIDS = (8, 16, 32, 64, 128)            # lattices of G x G oscillators per channel
CHANNELS = (1, 2, 4, 8, 16)
#: band mapping: 0 drives each of a lattice's G rows with its own mel band; 16 maps paper 02's 16
#: bands onto the rows. At 16 x 16 the two are the same, and it runs once.
MAPPINGS = (0, 16)
#: slimmed by the author on 2026-09-24 to one noise level and one input gain: 0 dB (paper 02 ran clean,
#: 0 and +5 dB; clean audio saturates the task) and gain 1 (paper 02 ran 1 and 2)
NOISES = (0.0,)
GAINS = (1.0,)
SEEDS = (0, 1, 2)
ANN_ARCHS = ("gru", "tcn", "cnn", "transformer", "s4d")
PHASE_FAMILIES = ("kuramoto", "sakaguchi", "harmonic2", "winfree")
AMPLITUDE_FAMILIES = ("sl", "sl-fixedamp")
SHAPES = ("torus", "cylinder", "sheet", "helix", "cube", "sphere")
CARRIER_GAIN = 32.0                     # the carrier pathway's calibrated gain in paper 02
CARRIER_NOISE = 0.0                     # paper 02 ran the carrier at 0 dB only
#: the carrier integrates 16,000 steps a clip; it is planned at every lattice until a GPU benchmark
#: (`plan benchmark`) prices it
CARRIER_GRIDS = GRIDS
#: the front end's longer real analysis window, as long as the zero-padded transform at that band count;
#: the own-band lattices at 64 and 128 run under both windows in the size, trained and quadrature tiers
LONG_WINDOW = {64: 1024, 128: 2048}
#: the unprojected (native) read is fitted where the features are small enough, as in paper 02's tier 4
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
    if arm.kind == "floor":
        return base, f"{base}@wholeclip"
    return (base, f"{base}+rate") if arm.kind == "field" else (base,)


def _net(tier: str, drive: str, noise, gain, seed, arm: Arm, task: str = "recognition", **kw) -> rn.Spec:
    return rn.Spec(tier, task, drive, noise, gain, seed, arm, native_sizes=_native(arm),
                   reads=reads_for(arm, task), **kw)


def _floor(tier: str, drive: str, noise, seed, grid: int, bands: int, window: int = 0,
           task: str = "recognition", **kw) -> rn.Spec:
    arm = Arm("floor", grid=grid, bands=bands, window=window)
    return rn.Spec(tier, task, drive, noise, None, seed, arm, reads=reads_for(arm, task), **kw)


def _reservoirs(channels: int, grid: int, bands: int, window: int = 0, bank: bool = True) -> tuple[Arm, ...]:
    """The coupled and uncoupled networks (and the state-matched bank) of one size."""
    arms = (Arm("field", channels=channels, grid=grid, bands=bands, window=window),
            Arm("field", channels=channels, grid=grid, bands=bands, window=window, severed=True))
    return arms + ((Arm("bank", channels=channels, grid=grid, bands=bands, window=window),) if bank else ())


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
    """The reference network, uncoupled network and state-matched bank at every lattice, channel count,
    band mapping and window, with the spectrogram-only baseline on the same rows; and the same arms on
    paper 02's order task at every lattice and band mapping."""
    for grid, bands, window in front_ends():
        for noise in NOISES:
            for seed in SEEDS:
                yield _floor("size", "envelope", noise, seed, grid, bands, window)
                for channels in CHANNELS:
                    for gain in GAINS:
                        for arm in _reservoirs(channels, grid, bands, window):
                            yield _net("size", "envelope", noise, gain, seed, arm)
    for grid, bands in lattices():
        for pair in pr.PAIRS:
            for noise in NOISES:
                for seed in SEEDS:
                    yield _floor("size", "envelope", noise, seed, grid, bands, task="order", pair=pair)
                    for channels in CHANNELS:
                        for gain in GAINS:
                            for arm in _reservoirs(channels, grid, bands):
                                yield _net("size", "envelope", noise, gain, seed, arm, "order", pair=pair)


def trained() -> Iterator[rn.Spec]:
    """The five trained baselines, each sized to every network of the size tier, on its rows."""
    for grid, bands, window in front_ends():
        for channels in CHANNELS:
            for arch in ANN_ARCHS:
                for noise in NOISES:
                    for seed in SEEDS:
                        arm = Arm("ann", arch=arch, channels=channels, grid=grid, bands=bands, window=window)
                        yield rn.Spec("trained", "recognition", "envelope", noise, None, seed, arm,
                                      reads=reads_for(arm, "recognition"))


def sequence() -> Iterator[rn.Spec]:
    """The digit-sequence task, 2, 3 and 4 digits, for the reference network, uncoupled network and
    state-matched bank at every lattice, channel count and band mapping, with the spectrogram-only
    baseline, which can tell which digits a sequence holds but not where."""
    for grid, bands in lattices():
        for length in pr.SEQUENCE_LENGTHS:
            for noise in NOISES:
                for seed in SEEDS:
                    yield _floor("sequence", "envelope", noise, seed, grid, bands, task="sequence", length=length)
                    for channels in CHANNELS:
                        for gain in GAINS:
                            for arm in _reservoirs(channels, grid, bands):
                                yield _net("sequence", "envelope", noise, gain, seed, arm, "sequence",
                                           length=length)


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
    for grid, bands, window in front_ends():
        for noise in NOISES:
            for seed in SEEDS:
                yield _floor("quadrature", "quadrature", noise, seed, grid, bands, window)
                for channels in CHANNELS:
                    for gain in GAINS:
                        for arm in _reservoirs(channels, grid, bands, window, bank=False):
                            yield _net("quadrature", "quadrature", noise, gain, seed, arm)


def carrier() -> Iterator[rn.Spec]:
    """The carrier pathway: the reference and uncoupled networks, the state-matched bank at the sample
    rate, and the pathway's own spectrogram-only baseline; 0 dB and gain 32, as in paper 02."""
    for grid, bands in lattices(CARRIER_GRIDS):
        for seed in SEEDS:
            yield _floor("carrier", "carrier", CARRIER_NOISE, seed, grid, bands)
            for channels in CHANNELS:
                for arm in _reservoirs(channels, grid, bands):
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
    """Every coupling function and geometry on the carrier pathway, at every lattice."""
    for grid, bands in lattices(CARRIER_GRIDS):
        for channels in CHANNELS:
            for family, shape in designs():
                if (family, shape) == REFERENCE:
                    continue
                for seed in SEEDS:
                    arm = Arm("field", physics=family, boundary=shape, channels=channels, grid=grid, bands=bands)
                    yield _net("design-carrier", "carrier", CARRIER_NOISE, CARRIER_GAIN, seed, arm)


TIERS = {"gate": gate, "size": size, "trained": trained, "sequence": sequence, "design": design,
         "quadrature": quadrature, "carrier": carrier, "design-quadrature": quadrature_design,
         "design-carrier": carrier_design}


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
ORDER02 = "tier1-order-envelope"
#: paper 02's projection tier: its record files are f"{PAPER02_PROJECTION}-{task}-{drive}"
PAPER02_PROJECTION = "projection"


def paper02_group(spec: rn.Spec) -> str | None:
    """The paper 02 record file that holds this spec's run, if paper 02 ran it.

    Paper 02 ran, at 16 x 16 with one band per row: the spectrogram-only
    baseline on every pathway; the reference network, its uncoupled copy and
    both banks (4 and 8 channels, the second being paper 03's state-matched
    bank at 8 channels) on the band-energy pathway; every coupling function
    and geometry at 4 channels (its tier 2); the trained baselines at 2,048
    parameters; and a diagonal of coupling functions and geometries on the
    quadrature and carrier pathways (its tier 3), with the 4-channel bank on
    the carrier. On the order task (its tier 1) it ran the spectrogram-only
    baseline, the reference network, its uncoupled copy and both banks. It did
    not run the digit-sequence task, or any window but its 512 samples.
    """
    a = spec.arm
    if spec.tier == "gate" or a.grid != 16 or a.n_bands != 16 or a.n_window != am.WINDOW:
        return None
    if spec.task == "sequence":
        return None
    if spec.task == "order":
        if a.kind == "ann" or (a.kind == "bank" and a.channels not in (4, 8)):
            return None
        if a.kind == "field" and (a.channels != 4 or (a.physics, a.boundary) != REFERENCE
                                  or (a.omega, a.damping, a.clamp) != ("random", 0.3, 1.0)):
            return None
        return ORDER02
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


def _paper02_record(group: str, run_id: str) -> dict | None:
    path = PAPER02_RECORD / f"{group}.json"
    return json.loads(path.read_text())["runs"].get(run_id) if path.exists() else None


def paper02_run(spec: rn.Spec) -> dict | None:
    """Paper 02's recorded run for a reused spec, with every cell tagged by its projection, or None if
    paper 02 has not recorded it (yet).

    Paper 02's earlier tiers recorded only the fixed projection and tagged
    nothing: an untagged cell is read as fixed, or as "none" where it was read
    unprojected. Its projection tier re-reads the reference reservoirs of its
    tier 1 (recognition and the order task) under both projections, tagging
    each cell "fixed", "seeded" or "none", under the same run identities; its
    seeded cells are added to the run here.
    """
    group = paper02_group(spec)
    rec = None if group is None else _paper02_record(group, spec.run_id())
    if rec is None:
        return None
    cells = [{**c, "projection": ro.projection_of(c, rec)} for c in rec["cells"]]
    again = _paper02_record(f"{PAPER02_PROJECTION}-{spec.task}-{spec.drive}", spec.run_id())
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


# ---------------------------------------------------------------------------
# Cost and memory, for ordering and for the estimate
# ---------------------------------------------------------------------------
#
# Milliseconds per clip for ONE channel of the reference network (Kuramoto,
# torus, band-energy pathway): simulating 61 frames, and computing the windowed
# statistics, on each device, with the coupling implementation `auto` picks
# there (models/phase.py). Measured on this project's Mac, an M1 Max with 64 GB:
#
#   cpu  one thread, 2026-09-23, while paper 02's runs held the other cores;
#        paper 02's tier 2 measured 16 x 16 x 4 at about 5 ms a clip, which
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
#: four fields instead of two; at 128 x 128 (FFT) the factors measured with FFT at 64 x 64.
SHAPE_COST = {"cpu": {"torus": 1.0, "cylinder": 1.9, "sheet": 2.4, "helix": 0.74, "cube": 1.66, "sphere": 1.19},
              "mps-fft": {"torus": 1.0, "cylinder": 1.06, "sheet": 1.66, "helix": 0.21, "cube": 2.43,
                          "sphere": 1.08}}
FAMILY_COST = {"cpu": {"kuramoto": 1.0, "sakaguchi": 1.2, "harmonic2": 2.6, "winfree": 2.2},
               "mps": {"kuramoto": 1.0, "sakaguchi": 1.0, "harmonic2": 2.0, "winfree": 1.0},
               "mps-fft": {"kuramoto": 1.0, "sakaguchi": 0.93, "harmonic2": 1.08, "winfree": 0.81}}
#: the Stuart-Landau cores: relative to the reference on the CPU (FFT, as in paper 02), and measured
#: outright on MPS (ms per channel and clip; dense up to 64 x 64, FFT at 128; 8 taken from 16, and 32
#: the geometric mean of its neighbours; the fixed-amplitude core assumed the same)
SL_COST = {"cpu": {"sl": 1.7, "sl-fixedamp": 0.9}}
SL_MS_MPS = {8: 0.063, 16: 0.063, 32: 0.31, 64: 1.51, 128: 5.04}
#: the bank, relative to the reference network (CPU), or measured (MPS; 8 and 32 interpolated)
BANK_SIM = {"cpu": 0.35}
BANK_MS_MPS = {8: 0.004, 16: 0.014, 32: 0.05, 64: 0.21, 128: 0.51}
BANK_FEAT = 0.5                         # the bank has one signal per state, the network two
QUADRATURE_COST = 1.3                   # measured on the CPU, assumed on MPS
CARRIER_STEPS = 16000 / 61              # the carrier integrates at the sample rate (extrapolated, both devices)
PROJECT_FLOPS = {"cpu": 150e9, "mps": 1.5e12}   # the projection's matrix products, measured
RANDN_PER_S = 48e6                      # Gaussian draws per second on one CPU thread (the matrices, every device)
TWO_THREADS = 1.65                      # measured speed-up drawing a channel's two matrices in two threads
RIDGE_S = 10.0                          # one read's ridge fits at every width under one projection, CPU float64
#: trained baselines, minutes to train at 2,048 clips and 30 epochs, measured per step at three budgets
ANN_MIN = {"cpu": {"gru": (0.6, 0.5, 3.8), "tcn": (0.2, 3.1, 22.7), "cnn": (0.3, 3.4, 22.7),
                   "transformer": (0.7, 0.9, 2.0), "s4d": (0.5, 16.2, 154.2)},
           "mps": {"gru": (0.98, 1.12, 1.09), "tcn": (0.28, 0.81, 0.25), "cnn": (0.17, 0.16, 0.19),
                   "transformer": (0.23, 0.29, 0.46), "s4d": (0.61, 0.97, 6.9)}}
ANN_BUDGETS = (2048, 65536, 524288)
CLIPS = rn.PRIMARY_SIZE + 6000
COST_DEVICES = ("cpu", "mps")


def _ann_minutes(arch: str, budget: int, device: str) -> float:
    """Log-log interpolation between the measured budgets (flat below the smallest)."""
    xs, ys = [math.log(b) for b in ANN_BUDGETS], [math.log(m) for m in ANN_MIN[device][arch]]
    x = math.log(max(budget, ANN_BUDGETS[0]))
    i = 0 if x <= xs[1] else 1
    return math.exp(ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i]))


def _sim_feat_ms(spec: rn.Spec, device: str) -> tuple[float, float]:
    """Simulation and statistics, ms per channel and clip, for the arm and pathway on a device."""
    a, grid = spec.arm, spec.arm.grid
    sim, feat = SIM_MS[device][grid], FEAT_MS[device][grid]
    if a.kind == "bank":
        sim = sim * BANK_SIM["cpu"] if device == "cpu" else BANK_MS_MPS[grid]
        feat *= BANK_FEAT
    elif a.physics in ("sl", "sl-fixedamp"):
        sim = sim * SL_COST["cpu"][a.physics] if device == "cpu" else SL_MS_MPS[grid]
    elif device == "cpu":
        sim *= SHAPE_COST["cpu"][a.boundary] * FAMILY_COST["cpu"][a.physics]
    elif grid > 64:                                      # FFT on MPS
        sim *= SHAPE_COST["mps-fft"][a.boundary] * FAMILY_COST["mps-fft"][a.physics]
    else:
        sim *= FAMILY_COST["mps"][a.physics]
    if spec.drive == "quadrature":
        sim *= QUADRATURE_COST
    if spec.drive == "carrier":
        sim, feat = sim * CARRIER_STEPS, feat * CARRIER_STEPS
    return sim, feat


def clips_and_frames(spec: rn.Spec) -> tuple[int, int]:
    """(clips a run reads, frames per clip): 8,048 of 61 for recognition, 4,096 of 147 for the order
    task, 4,096 of 147 to 284 for the digit-sequence task, and 16,000 samples per clip on the carrier."""
    if spec.task == "order":
        return pr.ORDER_TRAIN + pr.ORDER_TEST, pr.joined_frames(2)
    if spec.task == "sequence":
        return pr.SEQUENCE_TRAIN + pr.SEQUENCE_TEST, pr.joined_frames(spec.length)
    return CLIPS, (16000 if spec.drive == "carrier" else 61)


def features_per_state(arm: Arm, read: str) -> int:
    """Features per state for a read: 3 statistics per signal and window; a network has two signals per
    oscillator, and its rotation rates add two more per oscillator and window."""
    windows = am.WINDOWS["recognition"] if read.startswith("windowed") else 1
    per = 3 * windows * (2 if arm.kind == "field" else 1)
    return per + (2 * windows if read.endswith("+rate") else 0)


def seconds(spec: rn.Spec, device: str = "cpu") -> float:
    """Estimated seconds for one run: on one CPU thread (`cpu`) or on the M1 Max's GPU (`mps`).

    Both include what runs on the CPU either way: drawing the two projection
    matrices, and the ridge readout under each projection, for every read (and,
    on the digit-sequence task, every position). The frame counts and the
    number of clips scale the per-clip times measured on recognition; a streamed
    arm simulates its channels once per read.
    """
    a = spec.arm
    reads = spec.reads or tuple(am.reads(a, spec.task))
    positions = spec.length if spec.task == "sequence" else 1
    ridge = RIDGE_S * len(reads) * 2 * positions             # the fixed and the seeded projection
    clips, frames = clips_and_frames(spec)
    scale = frames / (16000 if spec.drive == "carrier" else 61)
    if a.kind == "ann":
        return 60 * _ann_minutes(a.arch, a.budget, device) * scale * clips / CLIPS + ridge
    if a.kind == "floor":
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
    if a.kind in ("floor", "ann"):
        return 1.0
    signals = (2 if a.kind == "field" else 1) * a.grid * a.grid
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
            and (not channels or s.arm.kind in ("floor",) or s.arm.channels in channels)]


def planned(names: list[str], grids=(), channels=()) -> list[rn.Spec]:
    specs = _select([s for name in names for s in TIERS[name]()], grids, channels)
    ids = [(s.group(), s.run_id()) for s in specs]
    if len(set(ids)) != len(ids):
        raise RuntimeError("two specs share a record address; the tier definitions are wrong")
    return specs


def pending(specs: list[rn.Spec]) -> list[rn.Spec]:
    """Specs not yet recorded here, and not recorded completely by paper 02. A reused spec whose paper 02
    run is missing (a paper 02 tier that has not run) or lacks the seeded projection is run here
    instead, and that run is the one reported; its fixed cells equal paper 02's (the reuse gate)."""
    done = {g: rn.recorded_ids(g) for g in {s.group() for s in specs}}
    return sorted((s for s in specs if s.run_id() not in done[s.group()] and not paper02_complete(s, paper02_run(s))),
                  key=seconds, reverse=True)


def estimate(names: list[str] | None = None) -> list[dict]:
    """Per tier and lattice: runs, runs reused from paper 02, hours on one CPU thread and on the M1 Max's
    GPU, the longest run on each, and the largest memory."""
    rows = []
    for name in names or list(TIERS):
        by = defaultdict(list)
        for s in TIERS[name]():
            by[s.arm.grid].append(s)
        for grid in sorted(by):
            specs = by[grid]
            todo = [s for s in specs if not reused(s)]
            row = {"tier": name, "grid": grid, "runs": len(specs), "reused": len(specs) - len(todo),
                   "peak_gb": max(memory_gb(s) for s in specs), "streamed": sum(s.streamed for s in specs)}
            for device in COST_DEVICES:
                row[f"{device}_hours"] = sum(seconds(s, device) for s in todo) / 3600
                row[f"{device}_longest_hours"] = max(seconds(s, device) for s in specs) / 3600
            rows.append(row)
    return rows


def _print_estimate(rows: list[dict]) -> None:
    print(f"{'tier':18s} {'grid':>5s} {'runs':>6s} {'reused':>6s} {'streamed':>8s} {'CPU-h':>9s} "
          f"{'MPS-h':>8s} {'longest CPU':>11s} {'longest MPS':>11s} {'peak GB':>8s}")
    totals = defaultdict(lambda: [0, 0, 0.0, 0.0])
    for r in rows:
        print(f"{r['tier']:18s} {r['grid']:5d} {r['runs']:6d} {r['reused']:6d} {r['streamed']:8d} "
              f"{r['cpu_hours']:9.1f} {r['mps_hours']:8.1f} {r['cpu_longest_hours']:10.2f}h "
              f"{r['mps_longest_hours']:10.2f}h {r['peak_gb']:8.1f}")
        t = totals[r["tier"]]
        t[0] += r["runs"]
        t[1] += r["reused"]
        t[2] += r["cpu_hours"]
        t[3] += r["mps_hours"]
    print()
    for tier, (runs, re, cpu, mps) in totals.items():
        print(f"{tier:18s} {runs:6d} runs, {re:4d} from paper 02, {cpu:8.0f} CPU-hours, {mps:7.1f} MPS-hours")
    print(f"{'all':18s} {sum(t[0] for t in totals.values()):6d} runs, {sum(t[1] for t in totals.values()):4d} "
          f"from paper 02, {sum(t[2] for t in totals.values()):8.0f} CPU-hours, "
          f"{sum(t[3] for t in totals.values()):7.1f} MPS-hours")


def cache_jobs(grids=GRIDS) -> list[tuple]:
    """Every row cache the tiers read: the bank's rows on the band-energy and quadrature pathways at each
    band count and window, and each order-task and digit-sequence set (its test set and every seed's
    training set) at each band count. The carrier's rows are never cached."""
    counts = sorted({16} | {g for g, b in lattices(grids) if b == 0})
    windows = {(g, w) for g, b, w in front_ends(grids) if w}
    codes = (0, *(seed + 1 for seed in SEEDS))
    jobs = [("rows", drive, noise, bands, pr.HOP_N_FFT) for drive in pr.CACHED_DRIVES for noise in NOISES
            for bands in counts]
    jobs += [("rows", drive, noise, g, w) for drive in pr.CACHED_DRIVES for noise in NOISES for g, w in sorted(windows)]
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
    """Build every row cache the tiers read, in parallel, skipping those that exist."""
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


def _work(spec: rn.Spec, device: str, threads: int) -> tuple[str, float]:
    """One run. A paper 02 cell run here (paper 02 has not recorded it completely) runs on the CPU,
    the only device on which it is bit-identical to paper 02's."""
    t0 = time.perf_counter()
    return rn.run(spec, "cpu" if reused(spec) else device, threads), time.perf_counter() - t0


def drive(names: list[str], workers: int, threads: int, device: str, dry_run: bool,
          grids=(), channels=()) -> None:
    device = resolve(device)
    specs = planned(names, grids, channels)
    todo = pending(specs)
    n_reused = sum(paper02_complete(s, paper02_run(s)) for s in specs)
    cost = "mps" if device == "mps" else "cpu"
    hours = sum(seconds(s, cost) for s in todo) / 3600
    print(f"=== {' + '.join(names)}: {len(specs)} runs planned, {n_reused} from paper 02, "
          f"{len(specs) - len(todo) - n_reused} recorded, {len(todo)} to run on {workers} worker(s) x "
          f"{threads} thread(s), device {device}; ~{hours:.0f} {cost.upper()}-hours")
    if dry_run:
        for s in todo[:12]:
            print(f"    {s.group()}/{s.run_id()}  ~{seconds(s, cost) / 60:.1f} min, ~{memory_gb(s):.1f} GB"
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
    r.add_argument("--device", default="auto", choices=DEVICES,
                   help="auto: CUDA if present, else Apple Silicon's GPU (mps), else the CPU; only cpu is "
                        "bit-identical to paper 02")
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

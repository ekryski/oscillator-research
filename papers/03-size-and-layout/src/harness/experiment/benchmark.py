"""Time one batch of each network on this machine's device, and extrapolate every run and tier from it.

    uv run python -m harness.experiment.plan benchmark --device cuda --out ../results/benchmark/cuda.json
    uv run python -m harness.experiment.plan benchmark --device mps --grids 8 16 --channels 1 4 --no-designs

The cost model in `plan.py` was measured on an M1 Max; the carrier, which
integrates 16,000 steps a clip, was extrapolated from the band-energy pathway
there, and a CUDA GPU was never measured. This measures what a run does on
whatever device it is given, for a small, representative fraction of its clips:

1. For each pathway (band-energy and carrier by default), lattice and channel
   count, the reference network and the state-matched bank are built exactly as
   a run builds them, and one batch of the run's own batch size (synthetic
   clips of the run's shape) is simulated, its instruments recorded and its
   statistics computed, after a warm-up. A streamed arm is timed on one of its
   channels, as it runs, and multiplied by its channel count. The uncoupled
   network costs what the coupled one does (its kernel is zero, not absent).
2. For each pathway and lattice, every other coupling function and geometry is
   timed the same way at one channel, as a factor on the reference network.
3. Once: the device's matrix-product rate (the projection), the CPU's
   Gaussian draws (the projection matrices) and one read's ridge fits (the
   CPU, float64); and, on the carrier, its front end, which runs on the CPU
   for every batch of every run and is never cached.

From these it extrapolates each run of every tier the way `plan.seconds` does
(clips, frames, reads, passes and positions), with the measured times in place
of the model's. The quadrature pathway, unless timed, is the band-energy
pathway times `plan.QUADRATURE_COST`; the trained baselines and the
spectrogram-only baseline are taken from the cost model on the M1 Max and
reported as modelled. The JSON report holds the measurements, every run's
estimated seconds, and each tier's hours by lattice.

Nothing here reads the digit bank or writes into the record.
"""

from __future__ import annotations

import json
import math
import platform
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from harness.experiment import arms as am
from harness.experiment import plan
from harness.experiment import readout as ro
from harness.experiment import run as rn
from harness.measurement.features import MAX_WIDTH
from harness.stimuli.filterbank import bandpass_rows
from harness.utils.device import describe, resolve

PATHWAYS = ("envelope", "carrier")
KINDS = ("field", "bank")
#: clips per timed batch at most (the run's own batch size is used below this)
MAX_CLIPS = 512


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _reset_peak(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def _free(device: str) -> None:
    """Return cached memory to the device, so one cell's allocations do not count against the next."""
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def _out_of_memory(e: Exception) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def _memory_gb(device: str) -> float | None:
    """Peak memory allocated on a CUDA device since the last reset; on MPS, what the driver holds now."""
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1e9
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return None


def synthetic_rows(drive: str, n: int, grid: int) -> torch.Tensor:
    """Front-end rows of a run's shape: 61 frames of band energies (or of quadrature pairs), or 16,000
    samples of band waveforms, made from noise. What a batch costs does not depend on its values."""
    gen = torch.Generator().manual_seed(0)
    if drive == "carrier":
        return bandpass_rows(torch.randn(n, 16000, generator=gen) * 0.05, grid)
    rows = torch.rand(n, 61, grid, generator=gen)
    if drive == "quadrature":
        phase = torch.rand(n, 61, grid, generator=gen) * 2 * math.pi
        return torch.stack((rows * phase.cos(), rows * phase.sin()), dim=-1)
    return rows


def time_arm(arm: am.Arm, drive: str, device: str, max_clips: int = MAX_CLIPS) -> dict:
    """Seconds per clip to simulate an arm, record a network's instruments and compute its statistics,
    for the whole arm (a streamed arm is timed on one channel and multiplied by its channels).

    If the run's batch does not fit in the device's memory, the batch is halved until it does, and
    the cell says so (`fits` false, `clips_timed` the batch that fitted): such a run would need a
    smaller batch on this device. A cell that does not fit even one clip is recorded without a time.
    """
    _free(device)
    gain = plan.CARRIER_GAIN if drive == "carrier" else 1.0
    spec = plan._net("benchmark", drive, 0.0, gain, 0, arm)
    rate = rn.CARRIER_RATE_HZ if drive == "carrier" else None
    model = am.build_frozen(arm, gain, 0, device, rate)
    one, sub, scale = arm, model, 1
    if spec.streamed:
        (one, sub), scale = am.channel(arm, model, 0), arm.channels
    run_batch = rn.batch_size(spec, device, states=one.states)
    n = max(1, min(max_clips, run_batch))
    rows = synthetic_rows(drive, n, arm.grid).to(device)
    tvalid = torch.full((n,), rows.shape[1], device=device)
    cell = {"run_batch": run_batch, "streamed": spec.streamed, "channels_timed": one.channels}

    def batch(r: torch.Tensor) -> None:
        with torch.no_grad():
            sig = am.frozen_signals(one, sub, r)
            if arm.kind == "field":
                am.field_instruments(sig, r, drive, one.channels, arm.grid)
            am.frozen_features(one, sig, tvalid[:len(r)], "recognition")

    fitted = n
    while True:
        try:
            batch(rows[:1])                            # warm-up: kernels, FFT plans, allocations
            _sync(device)
            _reset_peak(device)
            t0 = time.perf_counter()
            batch(rows[:fitted])
            _sync(device)
            took = time.perf_counter() - t0
            break
        except (RuntimeError, torch.OutOfMemoryError) as e:
            if not _out_of_memory(e):
                raise
            _free(device)
            if fitted == 1:
                return {**cell, "clips_timed": 0, "fits": False, "seconds_per_clip": None, "memory_gb": None}
            fitted //= 2
    out = {**cell, "clips_timed": fitted, "fits": fitted == n, "seconds_per_clip": took / fitted * scale,
           "memory_gb": _memory_gb(device)}
    model = sub = rows = None                          # drop the arm before freeing the cache
    _free(device)
    return out


def time_front_end(grid: int, n: int = 8) -> float:
    """Seconds per clip for the carrier's front end, on the CPU."""
    waves = torch.randn(n, 16000, generator=torch.Generator().manual_seed(1)) * 0.05
    bandpass_rows(waves[:1], grid)
    t0 = time.perf_counter()
    bandpass_rows(waves, grid)
    return (time.perf_counter() - t0) / n


def time_throughput(device: str) -> dict:
    """The projection's matrix-product rate on the device, the CPU's Gaussian draws per second, and one
    read's ridge fits at every width (8,048 clips, CPU float64)."""
    x = torch.randn(512, 16384, device=device)
    p = torch.randn(16384, MAX_WIDTH, device=device)
    x @ p
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(3):
        x @ p
    _sync(device)
    flops = 3 * 2 * 512 * 16384 * MAX_WIDTH / (time.perf_counter() - t0)
    del x, p
    t0 = time.perf_counter()
    torch.randn(4096, MAX_WIDTH, generator=torch.Generator().manual_seed(2))
    randn = 4096 * MAX_WIDTH / (time.perf_counter() - t0)
    gen = torch.Generator().manual_seed(3)
    feats = torch.randn(plan.CLIPS, MAX_WIDTH, generator=gen)
    labels = torch.randint(10, (plan.CLIPS,), generator=gen)
    n = rn.PRIMARY_SIZE
    t0 = time.perf_counter()
    for width in rn.WIDTHS:
        ro.ridge(feats[:n, :width], labels[:n], feats[n:, :width], labels[n:], 10)
    return {"matmul_flops": flops, "randn_per_s": randn, "ridge_s": time.perf_counter() - t0}


def _model_factor(spec: rn.Spec) -> float:
    """The cost model's factor for a coupling function and geometry on the M1 Max's GPU."""
    ref = replace(spec, arm=replace(spec.arm, physics=plan.REFERENCE[0], boundary=plan.REFERENCE[1]))
    return sum(plan._sim_feat_ms(spec, "mps")) / sum(plan._sim_feat_ms(ref, "mps"))


def run_seconds(spec: rn.Spec, m: dict) -> tuple[float, bool]:
    """(seconds, measured) for one run, from the measurements `m`; modelled for the trained and the
    spectrogram-only baselines, and for any arm the benchmark did not time."""
    a = spec.arm
    if a.kind in ("ann", "floor"):
        return plan.seconds(spec, "mps"), False
    kind = "bank" if a.kind == "bank" else "field"
    drive, factor = spec.drive, 1.0
    if (drive, a.grid, a.channels, kind) not in m["cells"] and drive == "quadrature":
        drive, factor = "envelope", plan.QUADRATURE_COST
    cell = m["cells"].get((drive, a.grid, a.channels, kind))
    if cell is None or cell["seconds_per_clip"] is None:
        return plan.seconds(spec, "mps"), False
    per_clip = cell["seconds_per_clip"] * factor
    if kind == "field" and (a.physics, a.boundary) != plan.REFERENCE:
        per_clip *= m["designs"].get((drive, a.grid, a.physics, a.boundary)) or _model_factor(spec)
    reads = spec.reads or tuple(am.reads(a, spec.task))
    positions = spec.length if spec.task == "sequence" else 1
    clips, frames = plan.clips_and_frames(spec)
    passes = len(reads) if spec.streamed else 1
    per_clip *= passes * frames / (16000 if spec.drive == "carrier" else 61)
    if spec.drive == "carrier":
        per_clip += m["front_end"][a.n_bands] * (a.channels * passes if spec.streamed else 1)
    t = m["throughput"]
    project = draws = 0.0
    for read in reads:
        native = plan.features_per_state(a, read) * a.grid * a.grid * a.channels
        project += 2 * 2 * native * min(MAX_WIDTH, native) / t["matmul_flops"]      # two projections
        draws += 2 * native * MAX_WIDTH / t["randn_per_s"] if native > MAX_WIDTH else 0.0
    if spec.streamed:
        draws /= plan.TWO_THREADS
    ridge = t["ridge_s"] * len(reads) * 2 * positions
    return clips * (per_clip + project) + draws + ridge, True


def extrapolate(m: dict, tiers=None) -> tuple[dict, list[dict]]:
    """Each tier's runs and hours by lattice (the modelled share apart), and every run's seconds. Runs
    taken from paper 02 are left out, as in `plan.estimate`."""
    by = defaultdict(lambda: defaultdict(lambda: {"runs": 0, "hours": 0.0, "modelled_runs": 0,
                                                  "modelled_hours": 0.0, "longest_hours": 0.0}))
    runs = []
    for name in tiers or plan.TIERS:
        for spec in plan.TIERS[name]():
            if plan.reused(spec):
                continue
            secs, measured = run_seconds(spec, m)
            row = by[name][spec.arm.grid]
            row["runs"] += 1
            row["hours"] += secs / 3600
            row["longest_hours"] = max(row["longest_hours"], secs / 3600)
            if not measured:
                row["modelled_runs"] += 1
                row["modelled_hours"] += secs / 3600
            runs.append({"tier": name, "run": f"{spec.group()}/{spec.run_id()}", "seconds": round(secs, 1),
                         "measured": measured})
    return {t: {str(g): v for g, v in sorted(rows.items())} for t, rows in by.items()}, runs


def benchmark(device: str = "auto", grids=plan.GRIDS, channels=plan.CHANNELS, pathways=PATHWAYS,
              designs: bool = True, max_clips: int = MAX_CLIPS, out: Path | None = None, log=print,
              throughput: dict | None = None) -> dict:
    """Measure, extrapolate, and write the report to `out` (if given). `throughput` skips measuring it."""
    device = resolve(device)
    env = {**describe(device), "torch": torch.__version__, "platform": platform.platform(),
           "cpu_threads": torch.get_num_threads()}
    log(f"=== benchmark: {env}")
    t_all = time.perf_counter()
    m = {"cells": {}, "designs": {}, "front_end": {}, "throughput": throughput or time_throughput(device)}
    t = m["throughput"]
    log(f"    projection {t['matmul_flops'] / 1e12:.2f} TFLOP/s, Gaussian draws {t['randn_per_s'] / 1e6:.0f} M/s, "
        f"one read's ridge fits {t['ridge_s']:.1f} s")
    for drive in pathways:
        for grid in grids:
            for bands in sorted({grid, 16}) if drive == "carrier" else ():
                if bands not in m["front_end"]:
                    m["front_end"][bands] = time_front_end(bands)
                    log(f"    carrier front end at {bands} bands: {m['front_end'][bands] * 1e3:.1f} ms per clip")
            for c in channels:
                for kind in (KINDS if drive != "quadrature" else ("field",)):
                    cell = time_arm(am.Arm(kind, channels=c, grid=grid), drive, device, max_clips)
                    m["cells"][(drive, grid, c, kind)] = cell
                    head = f"    {drive:10s} {grid:3d}x{grid:<3d} {c:2d} ch {kind:5s} "
                    if cell["seconds_per_clip"] is None:
                        log(head + "out of memory at one clip")
                        continue
                    log(head + f"{cell['seconds_per_clip'] * 1e3:11.2f} ms per clip ({cell['clips_timed']} clips"
                        + ("" if cell["fits"] else f"; the run's batch of {cell['run_batch']} does not fit")
                        + (", one channel" if cell["streamed"] else "")
                        + (f", {cell['memory_gb']:.1f} GB" if cell["memory_gb"] is not None else "") + ")")
            if not designs:
                continue
            ref = time_arm(am.Arm("field", grid=grid, channels=1), drive, device, max_clips)["seconds_per_clip"]
            if ref is None:
                continue
            families = plan.PHASE_FAMILIES + (() if drive == "quadrature" else plan.AMPLITUDE_FAMILIES)
            for family, shape in plan.designs(families):
                if (family, shape) == plan.REFERENCE:
                    continue
                arm = am.Arm("field", physics=family, boundary=shape, grid=grid, channels=1)
                t_design = time_arm(arm, drive, device, max_clips)["seconds_per_clip"]
                if t_design is not None:
                    m["designs"][(drive, grid, family, shape)] = t_design / ref
            log(f"    {drive:10s} {grid:3d}x{grid:<3d} designs: " + ", ".join(
                f"{fa}/{sh} {v:.2f}" for (d, g, fa, sh), v in m["designs"].items() if (d, g) == (drive, grid)))
    tiers, runs = extrapolate(m)
    totals = {name: {"runs": sum(r["runs"] for r in rows.values()), "hours": sum(r["hours"] for r in rows.values()),
                     "modelled_hours": sum(r["modelled_hours"] for r in rows.values())}
              for name, rows in tiers.items()}
    report = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"), "env": env, "benchmark_s": time.perf_counter() - t_all,
        "throughput": t, "front_end_s_per_clip": {str(g): v for g, v in m["front_end"].items()},
        "cells": [{"pathway": d, "grid": g, "channels": c, "kind": k, **v} for (d, g, c, k), v in m["cells"].items()],
        "design_factors": [{"pathway": d, "grid": g, "coupling": fa, "geometry": sh, "factor": v}
                           for (d, g, fa, sh), v in m["designs"].items()],
        "tiers": tiers, "totals": totals, "runs": runs,
    }
    for name, tot in totals.items():
        log(f"    {name:18s} {tot['runs']:6d} runs {tot['hours']:11.1f} hours on this device"
            + (f" ({tot['modelled_hours']:.1f} of them modelled)" if tot["modelled_hours"] else ""))
    log(f"    {'all':18s} {sum(v['runs'] for v in totals.values()):6d} runs "
        f"{sum(v['hours'] for v in totals.values()):11.1f} hours")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1) + "\n")
        log(f"=== wrote {out}")
    return report

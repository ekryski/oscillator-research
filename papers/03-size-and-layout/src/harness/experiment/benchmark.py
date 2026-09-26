"""Time one batch of each network on this machine's device, and extrapolate every run and tier from it.

    uv run python -m harness.experiment.plan benchmark --device cuda --out ../results/benchmark/cuda.json
    uv run python -m harness.experiment.plan benchmark --device mps --grids 8 16 --channels 1 4 --no-designs

The cost model in `plan.py` was measured on an M1 Max, and a CUDA GPU was
never measured. This measures what a run does on whatever device it is given,
for a small, representative fraction of its clips:

1. For each pathway (spectrogram and quadrature by default), lattice and channel
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
   CPU, float64).
4. The trained baselines' training steps (batches of 64) at 2,048, 65,536
   and 524,288 parameters, on the device and on the CPU, so the report says
   where each trains faster. Runs train them on the CPU unless told
   otherwise (`plan run --trained-device`), as paper 02 does.

From these it extrapolates each run of every tier the way `plan.seconds` does
(clips, frames, reads, passes and positions), with the measured times in place
of the model's. The quadrature pathway, unless timed, is the spectrogram
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
from harness.utils.device import describe, resolve

PATHWAYS = ("spectrogram", "quadrature")
KINDS = ("network", "bank")
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


def synthetic_rows(pathway: str, n: int, grid: int) -> torch.Tensor:
    """Front-end rows of a run's shape: 61 frames of band energies (or of quadrature pairs), made from
    noise. What a batch costs does not depend on its values."""
    gen = torch.Generator().manual_seed(0)
    rows = torch.rand(n, plan.FRAMES, grid, generator=gen)
    if pathway == "quadrature":
        phase = torch.rand(n, plan.FRAMES, grid, generator=gen) * 2 * math.pi
        return torch.stack((rows * phase.cos(), rows * phase.sin()), dim=-1)
    return rows


def time_arm(arm: am.Arm, pathway: str, device: str, max_clips: int = MAX_CLIPS) -> dict:
    """Seconds per clip to simulate an arm, record a network's instruments and compute its statistics,
    for the whole arm (a streamed arm is timed on one channel and multiplied by its channels).

    If the run's batch does not fit in the device's memory, the batch is halved until it does, and
    the cell says so (`fits` false, `clips_timed` the batch that fitted): such a run would need a
    smaller batch on this device. A cell that does not fit even one clip is recorded without a time.
    """
    _free(device)
    spec = plan._net("benchmark", pathway, 0.0, 1.0, 0, arm)
    model = am.build_untrained(arm, 1.0, 0, device)
    one, sub, scale = arm, model, 1
    if spec.streamed:
        (one, sub), scale = am.channel(arm, model, 0), arm.channels
    run_batch = rn.batch_size(spec, device, states=one.states)
    n = max(1, min(max_clips, run_batch))
    rows = synthetic_rows(pathway, n, arm.grid).to(device)
    tvalid = torch.full((n,), rows.shape[1], device=device)
    cell = {"run_batch": run_batch, "streamed": spec.streamed, "channels_timed": one.channels}

    def batch(r: torch.Tensor) -> None:
        with torch.no_grad():
            sig = am.untrained_signals(one, sub, r)
            if arm.kind == "network":
                am.network_instruments(sig, r, pathway, one.channels, arm.grid)
            am.untrained_features(one, sig, tvalid[:len(r)], "recognition")

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


#: (channels, lattice) of a network with each trained-baseline budget: 2,048, 65,536 and 524,288 parameters
TRAINED_SIZES = ((4, 16), (8, 64), (16, 128))


def time_trained(device: str, steps: int = 4, sizes=TRAINED_SIZES) -> list[dict]:
    """Seconds per training step (a batch of 64 clips of 61 frames) of each trained baseline at three
    budgets, on the device and on the CPU."""
    out = []
    gen = torch.Generator().manual_seed(4)
    for channels, grid in sizes:
        n = am.BATCH * (steps + 1)
        rows = torch.rand(n, 61, grid, generator=gen)
        tvalid = torch.full((n,), 61)
        labels = torch.randint(10, (n,), generator=gen)
        for arch in am.TRAINED:
            arm = am.Arm("trained", arch=arch, channels=channels, grid=grid)
            entry = {"arch": arch, "budget": arm.budget, "channels": channels, "grid": grid}
            for where in dict.fromkeys((device, "cpu")):
                am.train_baseline(arm, rows[:am.BATCH], tvalid[:am.BATCH], labels[:am.BATCH], "recognition", 0,
                                  epochs=1, device=where)                  # warm-up
                _sync(where)
                t0 = time.perf_counter()
                am.train_baseline(arm, rows, tvalid, labels, "recognition", 0, epochs=1, device=where)
                _sync(where)
                entry["cpu_s_per_step" if where == "cpu" else "device_s_per_step"] = (
                    (time.perf_counter() - t0) / (steps + 1))
            entry.setdefault("device_s_per_step", entry["cpu_s_per_step"])
            entry["device_speedup"] = entry["cpu_s_per_step"] / entry["device_s_per_step"]
            out.append(entry)
            _free(device)
    return out


def _model_factor(spec: rn.Spec) -> float:
    """The cost model's factor for a coupling function and geometry on the M1 Max's GPU."""
    ref = replace(spec, arm=replace(spec.arm, coupling=plan.REFERENCE[0], geometry=plan.REFERENCE[1]))
    return sum(plan._sim_feat_ms(spec, "mps")) / sum(plan._sim_feat_ms(ref, "mps"))


def run_seconds(spec: rn.Spec, m: dict) -> tuple[float, bool]:
    """(seconds, measured) for one run, from the measurements `m`; modelled for the trained and the
    spectrogram-only baselines, and for any arm the benchmark did not time."""
    a = spec.arm
    if a.kind in ("trained", "baseline"):
        return plan.seconds(spec, "mps"), False
    kind = "bank" if a.kind == "bank" else "network"
    pathway, factor = spec.pathway, 1.0
    if (pathway, a.grid, a.channels, kind) not in m["cells"] and pathway == "quadrature":
        pathway, factor = "spectrogram", plan.QUADRATURE_COST
    cell = m["cells"].get((pathway, a.grid, a.channels, kind))
    if cell is None or cell["seconds_per_clip"] is None:
        return plan.seconds(spec, "mps"), False
    per_clip = cell["seconds_per_clip"] * factor
    if kind == "network" and (a.coupling, a.geometry) != plan.REFERENCE:
        per_clip *= m["designs"].get((pathway, a.grid, a.coupling, a.geometry)) or _model_factor(spec)
    reads = spec.reads or tuple(am.reads(a, spec.task))
    positions = spec.length if spec.task == "sequence" else 1
    clips, frames = plan.clips_and_frames(spec)
    passes = len(reads) if spec.streamed else 1
    per_clip *= passes * frames / plan.FRAMES
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
              throughput: dict | None = None, trained: bool = True) -> dict:
    """Measure, extrapolate, and write the report to `out` (if given). `throughput` skips measuring it."""
    device = resolve(device)
    env = {**describe(device), "torch": torch.__version__, "platform": platform.platform(),
           "cpu_threads": torch.get_num_threads()}
    log(f"=== benchmark: {env}")
    t_all = time.perf_counter()
    m = {"cells": {}, "designs": {}, "throughput": throughput or time_throughput(device)}
    t = m["throughput"]
    log(f"    projection {t['matmul_flops'] / 1e12:.2f} TFLOP/s, Gaussian draws {t['randn_per_s'] / 1e6:.0f} M/s, "
        f"one read's ridge fits {t['ridge_s']:.1f} s")
    for pathway in pathways:
        for grid in grids:
            for c in channels:
                for kind in (KINDS if pathway != "quadrature" else ("network",)):
                    cell = time_arm(am.Arm(kind, channels=c, grid=grid), pathway, device, max_clips)
                    m["cells"][(pathway, grid, c, kind)] = cell
                    head = f"    {pathway:11s} {grid:3d}x{grid:<3d} {c:2d} ch {kind:5s} "
                    if cell["seconds_per_clip"] is None:
                        log(head + "out of memory at one clip")
                        continue
                    log(head + f"{cell['seconds_per_clip'] * 1e3:11.2f} ms per clip ({cell['clips_timed']} clips"
                        + ("" if cell["fits"] else f"; the run's batch of {cell['run_batch']} does not fit")
                        + (", one channel" if cell["streamed"] else "")
                        + (f", {cell['memory_gb']:.1f} GB" if cell["memory_gb"] is not None else "") + ")")
            if not designs:
                continue
            ref = time_arm(am.Arm("network", grid=grid, channels=1), pathway, device, max_clips)["seconds_per_clip"]
            if ref is None:
                continue
            couplings = plan.PHASE_COUPLINGS + (() if pathway == "quadrature" else plan.AMPLITUDE_COUPLINGS)
            for coupling, geometry in plan.designs(couplings):
                if (coupling, geometry) == plan.REFERENCE:
                    continue
                arm = am.Arm("network", coupling=coupling, geometry=geometry, grid=grid, channels=1)
                t_design = time_arm(arm, pathway, device, max_clips)["seconds_per_clip"]
                if t_design is not None:
                    m["designs"][(pathway, grid, coupling, geometry)] = t_design / ref
            log(f"    {pathway:11s} {grid:3d}x{grid:<3d} designs: " + ", ".join(
                f"{fa}/{sh} {v:.2f}" for (d, g, fa, sh), v in m["designs"].items() if (d, g) == (pathway, grid)))
    m["trained"] = time_trained(device) if trained else []
    for e in m["trained"]:
        log(f"    trained {e['arch']:11s} {e['budget']:7,d} parameters: {e['cpu_s_per_step'] * 1e3:8.1f} ms a step on "
            f"the CPU, {e['device_s_per_step'] * 1e3:8.1f} on {device} ({e['device_speedup']:.2f} times)")
    tiers, runs = extrapolate(m)
    totals = {name: {"runs": sum(r["runs"] for r in rows.values()), "hours": sum(r["hours"] for r in rows.values()),
                     "modelled_hours": sum(r["modelled_hours"] for r in rows.values())}
              for name, rows in tiers.items()}
    report = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"), "env": env, "benchmark_s": time.perf_counter() - t_all,
        "throughput": t,
        "cells": [{"pathway": d, "grid": g, "channels": c, "kind": k, **v} for (d, g, c, k), v in m["cells"].items()],
        "design_factors": [{"pathway": d, "grid": g, "coupling": fa, "geometry": sh, "factor": v}
                           for (d, g, fa, sh), v in m["designs"].items()],
        "trained": m["trained"], "tiers": tiers, "totals": totals, "runs": runs,
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

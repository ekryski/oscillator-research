"""Smoke run: one recorded run of every arm, coupling function, lattice geometry, pathway and task, rerun into
a scratch results folder and compared cell by cell with the paper's record. It checks that the code, on this
machine and device, reproduces what the record says. From papers/02-untrained-reservoirs/src:

    OSC_RESULTS_DIR=/tmp/smoke uv run python ../scripts/smoke_run.py --device mps --workers 3    # Apple GPU
    OSC_RESULTS_DIR=/tmp/smoke uv run python ../scripts/smoke_run.py --device cuda --workers 4   # CUDA GPU

The 28 runs take about ten minutes on an M1 Max. Differences of a few test clips are the devices' rounding.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from harness.experiment import plan
from harness.utils.paths import PAPER_ROOT

REAL = PAPER_ROOT / "results"


def chosen() -> list:
    out = []
    for s in plan.controls():
        a = s.arm
        if s.noise_db != 0.0 or s.seed != 0 or s.gain not in (None, 1.0):
            continue
        if s.task == "recognition" and a.coupling in ("kuramoto", "") and (a.kind != "trained" or s.sizes == (2048,)):
            out.append(s)
        elif s.task == "order" and tuple(s.pair) == (0, 6) and a.kind == "network" and a.coupled:
            out.append(s)
    for s in plan.design():
        a = s.arm
        if s.noise_db != 0.0 or s.seed != 0 or s.gain != 1.0 or a.restoring != 0.3 or a.ceiling != 1.0:
            continue
        if (a.geometry == "torus" and a.frequencies == "random") or (a.coupling == "kuramoto" and a.frequencies == "random") \
                or (a.coupling == "kuramoto" and a.geometry == "torus" and a.frequencies == "tonotopic"):
            out.append(s)
    for s in plan.cochlea():
        if s.noise_db == 0.0 and s.seed == 0 and s.gain == 1.0 and s.arm.coupling == "kuramoto" and s.arm.frequencies == "random":
            out.append(s)
    for s in plan.quadrature():
        a = s.arm
        if s.noise_db == 0.0 and s.seed == 0 and s.gain in (None, 1.0) and (a.kind == "baseline" or (
                a.coupling == "kuramoto" and a.geometry == "torus" and a.frequencies == "random")):
            out.append(s)
    return out


def _key(cell: dict) -> tuple:
    return cell["read"], cell["n_train"], cell["width"], cell.get("projection")


def compare(specs: list) -> None:
    scratch = Path(os.environ["OSC_RESULTS_DIR"])
    worst = 0
    for s in specs:
        new = json.loads((scratch / f"{s.group()}.json").read_text())["runs"].get(s.run_id())
        old = json.loads((REAL / f"{s.group()}.json").read_text())["runs"].get(s.run_id())
        if new is None or old is None:
            print(f"  MISSING {'new' if new is None else 'recorded'}: {s.group()}/{s.run_id()}")
            continue
        oc = {_key(c): c["acc"] for c in old["cells"]}
        diffs = [abs(c["acc"] - oc[_key(c)]) * new["n_test"] for c in new["cells"] if _key(c) in oc]
        top = max(diffs) if diffs else float("nan")
        worst = max(worst, top) if diffs else worst
        prim = [c for c in new["cells"] if c["n_train"] == 2048 and c["width"] == 192 and c["read"] in ("windowed", "pooled")]
        acc = f"{prim[0]['acc'] * 100:.2f}%" if prim else "-"
        print(f"  {s.group():34s} {s.run_id():72s} {acc:>8s}  max diff {top:4.0f} clips over {len(diffs)} cells"
              f"  (device {new['env']['device']} vs {old['env']['device']})")
    print(f"=== worst difference from the record: {worst:.0f} test clips")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    assert Path(os.environ["OSC_RESULTS_DIR"]).resolve() != REAL.resolve(), "point OSC_RESULTS_DIR at a scratch folder"
    specs = sorted(chosen(), key=plan.cost, reverse=True)
    print(f"=== smoke: {len(specs)} runs on {a.workers} worker(s), device {a.device}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex:
        futures = {ex.submit(plan._work, s, a.device, a.threads): s for s in specs}
        for i, f in enumerate(as_completed(futures), 1):
            s = futures[f]
            try:
                _, took = f.result()
                print(f"[{i:2d}/{len(specs)}] {time.perf_counter() - t0:6.0f}s {took:5.0f}s  {s.group()}/{s.run_id()}", flush=True)
            except Exception as e:                  # noqa: BLE001
                print(f"[{i:2d}/{len(specs)}] FAILED {s.group()}/{s.run_id()}: {e!r}", flush=True)
    compare(specs)
    print("=== done", flush=True)


if __name__ == "__main__":
    main()

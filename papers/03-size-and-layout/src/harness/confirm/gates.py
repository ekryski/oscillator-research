"""Integrity gates, checked before any result they guard is read.

    uv run python -m harness.confirm.gates reuse    # paper 02's 16 x 16 cells reproduce under this harness (CPU)
    uv run python -m harness.confirm.gates check    # every recorded zero-input cell reads chance

The reuse gate is what licenses taking cells from paper 02's record. It
re-runs a fixed sample of the paper 02 runs that paper 03 reuses, one from
each record file and arm kind, with this harness (exact kernel scaling, this
read), writes nothing into either record, and requires every cell's accuracy
and per-clip correctness to equal paper 02's. A sample whose paper 02 run has
not been recorded yet is reported as not checked, never as passed.

The zero-input gate reads the gate tier: with the input gain at 0 an arm
receives nothing, every clip's trajectory is the same, and every read must be
exactly chance, at every lattice and on both the in-memory and the streamed
read.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import torch

from harness.confirm import plan
from harness.confirm import run as rn
from harness.confirm.arms import Arm
from harness.confirm.readout import projection_of

CHANCE = 0.1
#: one reused run per record file and arm kind: (tier maker's spec fields)
REUSE_SAMPLE = (
    rn.Spec("size", "recognition", "envelope", 0.0, 1.0, 0, Arm("field")),
    rn.Spec("size", "recognition", "envelope", 0.0, 1.0, 1, Arm("field", severed=True)),
    rn.Spec("size", "recognition", "envelope", 0.0, 1.0, 2, Arm("bank", channels=8)),
    rn.Spec("size", "recognition", "envelope", 0.0, None, 0, Arm("floor"), reads=("windowed", "windowed@wholeclip")),
    rn.Spec("trained", "recognition", "envelope", 0.0, None, 1, Arm("ann", arch="gru"), reads=("windowed",)),
    rn.Spec("design", "recognition", "envelope", 0.0, 1.0, 0, Arm("field", physics="winfree", boundary="cube")),
    rn.Spec("design", "recognition", "envelope", 0.0, 1.0, 2, Arm("field", physics="sl", boundary="torus")),
    rn.Spec("design", "recognition", "envelope", 0.0, 1.0, 1, Arm("field", physics="sakaguchi", boundary="sheet")),
    rn.Spec("quadrature", "recognition", "quadrature", 0.0, 1.0, 0, Arm("field")),
    rn.Spec("carrier", "recognition", "carrier", 0.0, 32.0, 0, Arm("field")),
)


def reuse(threads: int = 4) -> dict:
    """Re-run the sample on the CPU; every accuracy and per-clip record must equal paper 02's.

    Always on the CPU: paper 02's record was made on CPUs, and only the CPU is
    bit-identical to it (a GPU's sums run in other orders).
    """
    device = "cpu"
    torch.set_num_threads(threads)
    report = {"sample": [], "passed": True, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    before = os.environ.get("OSC_RESULTS_DIR")
    with tempfile.TemporaryDirectory() as scratch:
        os.environ["OSC_RESULTS_DIR"] = scratch            # nothing is written into either record
        try:
            for spec in REUSE_SAMPLE:
                report["sample"].append(_check(spec, device))
                report["passed"] &= report["sample"][-1]["status"] != "FAIL"
        finally:
            if before is None:
                os.environ.pop("OSC_RESULTS_DIR", None)
            else:
                os.environ["OSC_RESULTS_DIR"] = before
    checked = [s for s in report["sample"] if s["status"] == "PASS"]
    report["passed"] &= bool(checked)
    _save("reuse", report)
    return report


def _check(spec: rn.Spec, device: str) -> dict:
    where = f"{plan.paper02_group(spec)}/{spec.run_id()}"
    old = plan.paper02_run(spec)
    if old is None:
        print(f"NOT CHECKED  {where}: paper 02 has not recorded it", flush=True)
        return {"run": where, "status": "NOT CHECKED"}
    new = rn.execute(spec, device)
    theirs = {(c["read"], c["n_train"], c["width"], projection_of(c, old)): c for c in old["cells"]}
    compared, mismatched = 0, []
    for c in new["cells"]:
        o = theirs.get((c["read"], c["n_train"], c["width"], c["projection"]))
        if o is None:
            continue
        compared += 1
        if o["acc"] != c["acc"] or ("correct" in o and "correct" in c and o["correct"] != c["correct"]):
            mismatched.append({"read": c["read"], "width": c["width"], "projection": c["projection"],
                               "paper02": o["acc"], "paper03": c["acc"]})
    status = "PASS" if compared and not mismatched else "FAIL"
    print(f"{status}  {where}: {compared - len(mismatched)}/{compared} cells identical", flush=True)
    return {"run": where, "status": status, "cells": compared, "mismatched": mismatched}


def check() -> dict:
    """Every recorded zero-input cell must read exactly chance."""
    report = {"g0_runs": 0, "passed": True, "off_chance": []}
    for path in sorted(rn.record_root().glob("gate-*.json")):
        for rid, r in json.loads(path.read_text())["runs"].items():
            report["g0_runs"] += 1
            for c in r["cells"]:
                if c["acc"] != CHANCE:
                    report["passed"] = False
                    report["off_chance"].append({"run": rid, "read": c["read"], "width": c["width"],
                                                 "acc": c["acc"]})
    report["passed"] &= report["g0_runs"] > 0
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['g0_runs']} zero-input run(s), "
          f"{len(report['off_chance'])} cell(s) off chance")
    _save("g0", report)
    return report


def _save(name: str, report: dict) -> None:
    path = rn.record_root() / "gates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = report
    path.write_text(json.dumps(data, indent=1) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="integrity gates")
    p.add_argument("gate", choices=("reuse", "check"))
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    ok = (reuse(a.threads) if a.gate == "reuse" else check())["passed"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

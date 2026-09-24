"""The integrity checks, run before the results they concern are read.

    uv run python -m harness.experiment.gates reuse    # paper 02's 16 x 16 cells reproduce under this harness (CPU)
    uv run python -m harness.experiment.gates check    # every recorded zero-input cell reads chance

The reuse check is what lets paper 03 take cells from paper 02's record. It
re-runs a fixed sample of the paper 02 runs that paper 03 reuses, one from
each record file and arm kind, with this harness (exact kernel scaling, this
read), writes nothing into either record, and compares every cell's accuracy
and per-clip correctness with paper 02's. A sample whose paper 02 run has not
been recorded yet is reported as not recorded.

The zero-input check reads the gate tier: with the input gain at 0 an arm
receives nothing, every clip's trajectory is the same, and every read should
be exactly chance, at every lattice and on both the in-memory and the
streamed read.

Both write what they found to gates.json, under `all_identical` and
`all_at_chance`, and exit non-zero if it is not so.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import torch

from harness.experiment import plan
from harness.experiment import run as rn
from harness.experiment.arms import Arm
from harness.experiment.readout import projection_of

CHANCE = 0.1
#: one reused run per record file and arm kind
REUSE_SAMPLE = (
    rn.Spec("size", "recognition", "spectrogram", 0.0, 1.0, 0, Arm("network")),
    rn.Spec("size", "recognition", "spectrogram", 0.0, 1.0, 1, Arm("network", coupled=False)),
    rn.Spec("size", "recognition", "spectrogram", 0.0, 1.0, 2, Arm("bank", channels=8)),
    rn.Spec("size", "recognition", "spectrogram", 0.0, None, 0, Arm("baseline"),
            reads=("windowed", "windowed@wholeclip")),
    rn.Spec("trained", "recognition", "spectrogram", 0.0, None, 1, Arm("trained", arch="gru"), reads=("windowed",)),
    rn.Spec("design", "recognition", "spectrogram", 0.0, 1.0, 0, Arm("network", coupling="winfree", geometry="cube")),
    rn.Spec("design", "recognition", "spectrogram", 0.0, 1.0, 2, Arm("network", coupling="stuart-landau")),
    rn.Spec("design", "recognition", "spectrogram", 0.0, 1.0, 1,
            Arm("network", coupling="kuramoto-sakaguchi", geometry="sheet")),
    rn.Spec("size", "order", "spectrogram", 0.0, 1.0, 0, Arm("network"), pair=(3, 7)),
    rn.Spec("quadrature", "recognition", "quadrature", 0.0, 1.0, 0, Arm("network")),
    rn.Spec("carrier", "recognition", "carrier", 0.0, 32.0, 0, Arm("network")),
)


def reuse(threads: int = 4) -> dict:
    """Re-run the sample on the CPU and compare every accuracy and per-clip record with paper 02's.

    Always on the CPU: paper 02's record was made on CPUs, and only the CPU is
    bit-identical to it (a GPU's sums run in other orders).
    """
    device = "cpu"
    torch.set_num_threads(threads)
    report = {"sample": [], "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    before = os.environ.get("OSC_RESULTS_DIR")
    with tempfile.TemporaryDirectory() as scratch:
        os.environ["OSC_RESULTS_DIR"] = scratch            # nothing is written into either record
        try:
            for spec in REUSE_SAMPLE:
                report["sample"].append(_compare(spec, device))
        finally:
            if before is None:
                os.environ.pop("OSC_RESULTS_DIR", None)
            else:
                os.environ["OSC_RESULTS_DIR"] = before
    compared = [s for s in report["sample"] if s["status"] != "not recorded"]
    report["all_identical"] = bool(compared) and all(s["status"] == "identical" for s in compared)
    _save("reuse", report)
    return report


def _compare(spec: rn.Spec, device: str) -> dict:
    where = f"{plan.paper02_group(spec)}/{spec.run_id()}"
    old = plan.paper02_run(spec)
    if old is None:
        print(f"not recorded  {where}: paper 02 has not recorded it", flush=True)
        return {"run": where, "status": "not recorded"}
    new = rn.execute(spec, device)
    theirs = {(c["read"], c["n_train"], c["width"], projection_of(c, old)): c for c in old["cells"]}
    compared, differing = 0, []
    for c in new["cells"]:
        o = theirs.get((c["read"], c["n_train"], c["width"], c["projection"]))
        if o is None:
            continue
        compared += 1
        if o["acc"] != c["acc"] or ("correct" in o and "correct" in c and o["correct"] != c["correct"]):
            differing.append({"read": c["read"], "width": c["width"], "projection": c["projection"],
                              "paper02": o["acc"], "paper03": c["acc"]})
    status = "identical" if compared and not differing else "differs"
    print(f"{status:12s}  {where}: {compared - len(differing)}/{compared} cells identical", flush=True)
    return {"run": where, "status": status, "cells": compared, "differing": differing}


def check() -> dict:
    """Every recorded zero-input cell should read exactly chance."""
    report = {"g0_runs": 0, "off_chance": []}
    for path in sorted(rn.record_root().glob("gate-*.json")):
        for rid, r in json.loads(path.read_text())["runs"].items():
            report["g0_runs"] += 1
            for c in r["cells"]:
                if c["acc"] != CHANCE:
                    report["off_chance"].append({"run": rid, "read": c["read"], "width": c["width"],
                                                 "acc": c["acc"]})
    report["all_at_chance"] = report["g0_runs"] > 0 and not report["off_chance"]
    print(f"{report['g0_runs']} zero-input run(s), {len(report['off_chance'])} cell(s) off chance")
    _save("g0", report)
    return report


def _save(name: str, report: dict) -> None:
    path = rn.record_root() / "gates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = report
    path.write_text(json.dumps(data, indent=1) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="the integrity checks")
    p.add_argument("check", choices=("reuse", "check"))
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    ok = reuse(a.threads)["all_identical"] if a.check == "reuse" else check()["all_at_chance"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

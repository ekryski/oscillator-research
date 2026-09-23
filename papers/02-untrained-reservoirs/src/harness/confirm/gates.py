"""Integrity gates, checked before any scored result is read.

    uv run python -m harness.confirm.gates legacy   # the exploratory record reproduces
    uv run python -m harness.confirm.gates check    # the recorded g = 0 cells read chance

The legacy gate re-runs a fixed sample of exploratory runs with the exploratory
harness, from the configurations the record stored, and compares every field.
Accuracies must match exactly. Continuous diagnostics (ridge margins, locking
instruments) are reported with their largest difference, because they move in
the last digits between library versions; no verdict reads them. The sample
spans a phase core, an amplitude core, a designed and a quadrature cell, the
order task and a trained network, so each code path the paper relies on is
exercised once.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from harness.confirm import run as rn
from harness.utils.paths import PAPER_ROOT

EXPLORATORY = PAPER_ROOT / "results"
LEGACY_SAMPLE = (
    ("envelope/matrix-kuramoto.json", "torus-random-lam0.3-clamp1-0db-g2"),
    ("envelope/matrix-sl.json", "torus-random-lam0.3-clamp1-5db-g1"),
    ("envelope/matrix-winfree.json", "cube-designed-lam0.1-clamp0.5-5db-g2"),
    ("quadrature/matrix-kuramoto.json", "cube-designed-lam0.1-clamp0.5-0db-g2"),
    ("envelope/order.json", "pair37-kuramoto"),
    ("baselines/conventional.json", "gru-0db-frozenprobe"),
)
#: row fields that are accuracies (must match exactly) and bookkeeping (ignored)
EXACT_PREFIXES = ("ridge_acc", "frz_acc", "lam", "arm", "seed", "clamp", "damping", "healthy")
IGNORED = ("wall_sec", "sec_ep")
CHANCE = 0.1


def legacy(threads: int = 4) -> dict:
    """Re-run the sample; return a report and write it to the gate record."""
    torch.set_num_threads(threads)
    report = {"sample": [], "passed": True, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    before = os.environ.get("OSC_RESULTS_DIR")
    with tempfile.TemporaryDirectory() as scratch:
        os.environ["OSC_RESULTS_DIR"] = scratch          # the re-runs never write into the committed record
        try:
            _rerun(report, threads)
        finally:
            if before is None:
                os.environ.pop("OSC_RESULTS_DIR", None)
            else:
                os.environ["OSC_RESULTS_DIR"] = before
    _save("legacy", report)
    return report


def _rerun(report: dict, threads: int) -> None:
    import argparse as ap
    from harness import runner
    for group, run_id in LEGACY_SAMPLE:
        entry = json.loads((EXPLORATORY / group).read_text())["runs"][run_id]
        cfg = dict(entry["config"])
        kind = cfg.pop("kind", "matrix")
        cfg.pop("n_classes", None)
        # arguments added after a run was recorded take the defaults it ran under
        base = vars(runner.parse_args(["--task", cfg["task"]]))
        args = ap.Namespace(**{**base, **cfg})
        args.kind, args.threads = kind, threads
        rows = runner.run_matrix(args)
        for new, old in zip(rows, entry["rows"]):
            keys = [k for k in old if k not in IGNORED]
            exact = [k for k in keys if k.startswith(EXACT_PREFIXES)]
            mismatched = [k for k in exact if old[k] != new.get(k)]
            drift = {k: abs(old[k] - new[k]) for k in keys if k not in exact
                     and isinstance(old[k], float) and isinstance(new.get(k), float)}
            ok = not mismatched and len(rows) == len(entry["rows"])
            report["passed"] &= ok
            report["sample"].append({"group": group, "run": run_id, "arm": old["arm"], "seed": old["seed"],
                                     "exact_fields": len(exact), "mismatched": mismatched,
                                     "largest_drift": max(drift.values(), default=0.0),
                                     "drift_fields": sorted(k for k, v in drift.items() if v > 0),
                                     "passed": ok})
            print(f"{'PASS' if ok else 'FAIL'}  {group} {run_id} {old['arm']} s{old['seed']}: "
                  f"{len(exact) - len(mismatched)}/{len(exact)} accuracies exact, "
                  f"largest diagnostic drift {max(drift.values(), default=0.0):.1e}", flush=True)


def check() -> dict:
    """Every recorded g = 0 cell must read exactly chance."""
    runs = rn.load_group("gate-recognition-envelope")["runs"]
    zero = {rid: r for rid, r in runs.items() if "/g0/" in rid and not rid.endswith("clipspan")}
    report = {"g0_runs": len(zero), "passed": bool(zero), "off_chance": []}
    for rid, r in zero.items():
        for c in r["cells"]:
            if c["acc"] != CHANCE:
                report["passed"] = False
                report["off_chance"].append({"run": rid, "read": c["read"], "width": c["width"], "acc": c["acc"]})
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {len(zero)} g = 0 run(s), "
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
    p.add_argument("gate", choices=("legacy", "check"))
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    ok = (legacy(a.threads) if a.gate == "legacy" else check())["passed"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

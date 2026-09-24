"""The integrity check, run before any result is read.

    uv run python -m harness.experiment.gates check    # every recorded g = 0 cell reads chance

With no input a reservoir's state carries nothing about the clip, so every
cell of a zero-gain run must read exactly chance. Anything else would mean
the read or the data leaks information around the reservoir.
"""

from __future__ import annotations

import argparse
import json

from harness.experiment import run as rn

CHANCE = 0.1


def check() -> dict:
    """Every recorded g = 0 cell should read exactly chance."""
    runs = rn.load_group("gate-recognition-spectrogram")["runs"]
    zero = {rid: r for rid, r in runs.items() if "/g0/" in rid and not rid.endswith("clipspan")}
    report = {"g0_runs": len(zero), "all_at_chance": bool(zero), "off_chance": []}
    for rid, r in zero.items():
        for c in r["cells"]:
            if c["acc"] != CHANCE:
                report["all_at_chance"] = False
                report["off_chance"].append({"run": rid, "read": c["read"], "width": c["width"], "acc": c["acc"]})
    print(f"{len(zero)} g = 0 run(s), {len(report['off_chance'])} cell(s) off chance")
    _save("g0", report)
    return report


def _save(name: str, report: dict) -> None:
    path = rn.record_root() / "gates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = report
    path.write_text(json.dumps(data, indent=1) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="the integrity check")
    p.add_argument("gate", choices=("check",))
    p.parse_args(argv)
    ok = check()["all_at_chance"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Where the harness reads data from and writes results to.

Every path is anchored on this package's own location, so commands work from
any working directory:

    <paper>/src/data/        the bank and row caches (gitignored)
    <paper>/results/         the record (committed)
    <paper>/resources/       figures that ship with the paper

Both roots accept an environment override, `OSC_DATA_DIR` and
`OSC_RESULTS_DIR`, for running against a scratch disk or reproducing into a
fresh tree without touching the committed one. The results override is read
per call, so a test or a shell can set it late.
"""

from __future__ import annotations

import os
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]  # harness/utils/paths.py -> src/
PAPER_ROOT = SRC_ROOT.parent
#: paper 02, whose record paper 03 reuses at 16 x 16 and 4 channels (read, never written)
PAPER02_ROOT = PAPER_ROOT.parent / "02-untrained-reservoirs"

DATA_DIR = Path(os.environ.get("OSC_DATA_DIR", SRC_ROOT / "data"))
RESULTS_DIR = PAPER_ROOT / "results"
RESOURCES_DIR = PAPER_ROOT / "resources"

CACHE_DIR = DATA_DIR / "cache"
AUDIOMNIST_DIR = DATA_DIR / "AudioMNIST" / "data"
FIGURES_DIR = RESOURCES_DIR / "figures"


def results_root() -> Path:
    """Where the record lives, resolved per call so `OSC_RESULTS_DIR` works from a test or mid-process."""
    return Path(os.environ.get("OSC_RESULTS_DIR", RESULTS_DIR))

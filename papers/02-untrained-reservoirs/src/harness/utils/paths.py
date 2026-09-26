"""Where the harness reads data from and writes results to.

Every path is anchored on this package's own location, so commands work from
any working directory:

    <paper>/src/data/        corpora and derived stimulus banks (gitignored)
    <paper>/results/         one directory per experiment run (committed)
    <paper>/resources/       figures and audio that ship with the paper

Both roots accept an environment override — `OSC_DATA_DIR` and
`OSC_RESULTS_DIR` — for running against a scratch disk or reproducing into a
fresh tree without touching the committed one.
"""

from __future__ import annotations

import os
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]  # harness/utils/paths.py -> src/
PAPER_ROOT = SRC_ROOT.parent

DATA_DIR = Path(os.environ.get("OSC_DATA_DIR", SRC_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("OSC_RESULTS_DIR", PAPER_ROOT / "results"))
RESOURCES_DIR = PAPER_ROOT / "resources"

CACHE_DIR = DATA_DIR / "cache"
AUDIOMNIST_DIR = DATA_DIR / "AudioMNIST" / "data"
FIGURES_DIR = RESOURCES_DIR / "figures"
AUDIO_DIR = RESOURCES_DIR / "audio"

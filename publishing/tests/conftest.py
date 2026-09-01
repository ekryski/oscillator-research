"""Put publishing/lib on the path — these modules are scripts, not a package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

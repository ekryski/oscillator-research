"""The harness for paper 02: spoken-digit recognition with untrained oscillator networks.

| package        | what lives there                                                        |
|----------------|-------------------------------------------------------------------------|
| `utils`        | the fixed constants; where data and results live                       |
| `stimuli`      | sound in: the clips, the fixed front ends, the injection path          |
| `models`       | the arms: the oscillator networks and their cores, the leaky-integrator |
|                | banks and the trained baselines; geometries get a subpackage            |
| `measurement`  | the shared statistics and projection, and the drive's phase             |
| `experiment`   | the study: its data protocol, arms, read, runs, experiments and summary |
"""

from harness.measurement import *  # noqa: F401,F403
from harness.measurement import __all__ as _measurement
from harness.models import *  # noqa: F401,F403
from harness.models import __all__ as _models
from harness.stimuli import *  # noqa: F401,F403
from harness.stimuli import __all__ as _stimuli
from harness.utils import *  # noqa: F401,F403
from harness.utils import __all__ as _utils

__all__ = sorted({*_measurement, *_models, *_stimuli, *_utils})

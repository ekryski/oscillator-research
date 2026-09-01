"""Fixed constants of the pre-registered protocol.

The harness has no audio sample rate of its own: one "frame" is one core
integration frame (substeps=1, dt=0.1), so synthetic-stimulus frequencies are
quoted in cycles/frame. Speech tasks resample to 16 kHz upstream and enter
through the hop frontend (62.5 fps) or the sample-rate carrier band-split.
"""

import math

TWO_PI = 2 * math.pi

# --- fixed frontend constants (part of the pre-registered task definition) ---
F_LO = 0.006  # lowest filterbank edge, cycles/frame
BANDS_PER_OCTAVE = 4  # 16 rows span 4 octaves: F_LO .. F_LO * 2^4
GAIN = 2.0  # default drive multiplier
WARMUP_FRAMES = 16  # frames excluded from features/PLV (integrator settle)
PROBE_SCALE = 4.0  # fixed logit temperature for the frozen random probe
PLV_LOCK_THRESH = 0.5  # an oscillator with PLV above this counts as entrained
N_SETTLE = 32  # settle-read: drive-free continuation frames after the clip

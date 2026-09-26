"""Fixed constants of the protocol, carried over from paper 02 unchanged.

One "frame" is one core integration frame (substeps=1, dt=0.1). Speech enters
through the hop front end (62.5 frames per second).
"""

import math

TWO_PI = 2 * math.pi

WARMUP_FRAMES = 16      # frames excluded from every read (the integrator's settle)
PLV_LOCK_THRESH = 0.5   # an oscillator whose locking value to its drive exceeds this counts as entrained
#: the canonical values of the coupling functions' extra parameters (paper 02)
ALPHA = math.pi / 4     # Kuramoto-Sakaguchi phase lag
BETA = 0.5              # second-harmonic weight

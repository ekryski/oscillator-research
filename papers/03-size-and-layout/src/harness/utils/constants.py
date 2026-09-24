"""Fixed constants of the protocol, carried over from paper 02 unchanged.

One "frame" is one core integration frame (substeps=1, dt=0.1). Speech enters
through the hop front end (62.5 frames per second) or, on the carrier pathway,
at the 16 kHz sample rate.
"""

import math

TWO_PI = 2 * math.pi

F_LO = 0.006            # lowest log-spaced band edge, cycles/frame
OCTAVES = 4             # the log-spaced bands span four octaves at every band count
BANDS_PER_OCTAVE = 4    # at 16 bands, as in paper 02; G bands have G / 4 per octave
WARMUP_FRAMES = 16      # frames excluded from every read (the integrator's settle)
PLV_LOCK_THRESH = 0.5   # an oscillator whose locking value to its drive exceeds this counts as entrained
#: the canonical values of the coupling functions' extra parameters (paper 02)
ALPHA = math.pi / 4     # Kuramoto-Sakaguchi phase lag
BETA = 0.5              # second-harmonic weight

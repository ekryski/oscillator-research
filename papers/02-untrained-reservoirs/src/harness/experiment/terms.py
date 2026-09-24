"""The paper's terms for the record's labels, in one place.

The record's labels follow the paper's glossary (Appendix A) in short form:
`baseline`, `coupled-kuramoto-torus-random-restoring0.3-ceiling1`, `bank-state`,
`trained-gru`, the `spectrogram` pathway. Everything people read (tables,
figures, the summary report) names things in full with the words defined here,
so no two places can drift apart. The same mapping is tabulated in src/README.md.
"""

from __future__ import annotations

import re

#: arms named by label
ARMS = {
    "baseline": "spectrogram-only baseline",
    "bank-state": "leaky-integrator bank, state-matched",
    "bank-width": "leaky-integrator bank, width-matched",
    "trained-gru": "GRU",
    "trained-tcn": "TCN",
    "trained-cnn": "CNN",
    "trained-transformer": "transformer",
    "trained-s4d": "S4D",
}

#: the design factors, as the Arm fields name them
FACTORS = {"coupling": "coupling function", "geometry": "lattice geometry", "frequencies": "natural frequencies",
           "restoring": "restoring strength", "ceiling": "coupling ceiling"}

#: a factor's levels
LEVELS = {
    "kuramoto": "Kuramoto", "kuramoto-sakaguchi": "Kuramoto–Sakaguchi", "second-harmonic": "second harmonic",
    "winfree": "Winfree", "stuart-landau": "Stuart–Landau", "stuart-landau-fixed": "Stuart–Landau, fixed amplitude",
    "random": "random", "tonotopic": "tonotopic", "identical": "identical",
}

#: the input pathways
PATHWAYS = {"spectrogram": "spectrogram", "quadrature": "quadrature"}

#: the Tier 1 reference configuration of the coupled network
REFERENCE = {"coupling": "kuramoto", "geometry": "torus", "frequencies": "random", "restoring": 0.3, "ceiling": 1.0}

_NETWORK = re.compile(r"^(coupled|uncoupled)-"
                      r"(?P<coupling>kuramoto-sakaguchi|second-harmonic|stuart-landau-fixed|stuart-landau|[a-z]+)-"
                      r"(?P<geometry>[a-z-]+?)-(?P<frequencies>random|tonotopic|identical)-restoring(?P<restoring>[0-9.]+)"
                      r"-ceiling(?P<ceiling>[0-9.]+)(?:-ch(?P<channels>\d+))?$")


def level(value) -> str:
    """A factor level in the paper's words: 'second-harmonic' -> 'second harmonic', 0.3 -> '0.3'."""
    return LEVELS.get(value, value) if isinstance(value, str) else f"{value:g}"


def arm(label: str) -> str:
    """The paper's name for a record label.

    The Tier 1 configuration of the oscillator networks is named plainly; any
    other configuration names only the factors where it departs from it.
    """
    if label in ARMS:
        return ARMS[label]
    m = _NETWORK.match(label)
    if not m:
        return label
    name = "uncoupled oscillator network" if label.startswith("uncoupled") else "coupled oscillator network"
    parts = [f"{level(m['coupling'])}" if m["coupling"] != REFERENCE["coupling"] else "",
             m["geometry"] if m["geometry"] != REFERENCE["geometry"] else "",
             f"{level(m['frequencies'])} ω" if m["frequencies"] != REFERENCE["frequencies"] else "",
             f"λ {m['restoring']}" if float(m["restoring"]) != REFERENCE["restoring"] else "",
             f"ceiling {m['ceiling']}" if float(m["ceiling"]) != REFERENCE["ceiling"] else "",
             f"{m['channels']} channels" if m["channels"] else ""]
    parts = [p for p in parts if p]
    return name + (f" ({', '.join(parts)})" if parts else "")


def read(name: str) -> str:
    """A read's qualifier: '' for the primary read, else what differs from it."""
    out = ["whole clip"] if "@wholeclip" in name else []
    return ", ".join(out + (["with rotation rates"] if "+rate" in name else []))

"""The paper's terms for the record's labels, in one place.

The record keeps the labels the registration froze (`floor`, `field-...`,
`bank-c4`, `envelope`), because they are its run identities. Everything people
read (tables, figures, the summary report) names things with the terms the
paper's glossary defines (Appendix A), and gets them from here, so no two
places can drift apart. The same mapping is tabulated in src/README.md.
"""

from __future__ import annotations

import re

#: arms the registration names by kind
ARMS = {
    "floor": "spectrogram-only baseline",
    "bank-c4": "leaky-integrator bank, state-matched",
    "bank-c8": "leaky-integrator bank, width-matched",
    "ann-gru": "GRU",
    "ann-tcn": "TCN",
    "ann-cnn": "CNN",
    "ann-transformer": "transformer",
    "ann-s4d": "S4D",
}

#: the design factors, as the Arm fields name them
FACTORS = {"physics": "coupling function", "boundary": "lattice geometry", "omega": "natural frequencies",
           "damping": "restoring strength", "clamp": "coupling ceiling", "channels": "channels"}

#: a factor's levels
LEVELS = {
    "kuramoto": "Kuramoto", "sakaguchi": "Kuramoto–Sakaguchi", "harmonic2": "second harmonic",
    "winfree": "Winfree", "sl": "Stuart–Landau", "sl-fixedamp": "Stuart–Landau, fixed amplitude",
    "random": "random", "designed": "tonotopic", "uniform": "identical",
}

#: the input pathways, as the record names them
PATHWAYS = {"envelope": "band-energy", "quadrature": "quadrature", "carrier": "carrier"}

#: the Tier 1 reference configuration of the coupled network
REFERENCE = {"physics": "kuramoto", "boundary": "torus", "omega": "random", "damping": 0.3, "clamp": 1.0}

_LATTICE = re.compile(r"^(?P<base>.*?)(?:-(?P<grid>\d+)x\d+)?(?:-(?P<bands>\d+)bands)?$")
_OSCILLATORS = re.compile(r"^(field|severed)-(?P<physics>sl-fixedamp|[a-z0-9]+)-(?P<boundary>[a-z]+)-"
                          r"(?P<omega>[a-z]+)-lam(?P<damping>[0-9.]+)-clamp(?P<clamp>[0-9.]+)(?:-c(?P<channels>\d+))?$")


def level(value) -> str:
    """A factor level in the paper's words: 'designed' -> 'tonotopic', 0.3 -> '0.3'."""
    return LEVELS.get(value, value) if isinstance(value, str) else f"{value:g}"


def _lattice(grid: str | None, bands: str | None) -> list[str]:
    """What a label's lattice suffix says: its size, and how its rows are driven if not one band each."""
    out = [f"{grid} × {grid} lattice"] if grid else []
    if bands:
        out.append(f"{bands} mel bands mapped onto {grid} rows")
    elif grid:
        out.append(f"{grid} mel bands")
    return out


def _channels(n: int) -> str:
    return f"{n} channel" + ("" if n == 1 else "s")


def arm(label: str, tier: str | None = None) -> str:
    """The paper's name for a record label.

    The Tier 1 configuration of the oscillator networks is named plainly; any
    other configuration names only the factors where it departs from it. In
    Tier 4 every bank is state-matched to the network with its channel count,
    so `bank-c8` there is not the width-matched bank of the other tiers.
    """
    lat = _LATTICE.match(label)
    base, extra = lat["base"], _lattice(lat["grid"], lat["bands"])
    bank = re.fullmatch(r"bank-c(\d+)", base)
    if bank and (tier == "tier4" or base not in ARMS):
        return f"leaky-integrator bank, state-matched ({', '.join([_channels(int(bank[1])), *extra])})"
    if base in ARMS:
        return ARMS[base] + (f" ({', '.join(extra)})" if extra else "")
    m = _OSCILLATORS.match(base)
    if not m:
        return label
    name = "uncoupled oscillator network" if label.startswith("severed") else "coupled oscillator network"
    parts = [f"{level(m['physics'])}" if m["physics"] != REFERENCE["physics"] else "",
             m["boundary"] if m["boundary"] != REFERENCE["boundary"] else "",
             f"{level(m['omega'])} ω" if m["omega"] != REFERENCE["omega"] else "",
             f"λ {m['damping']}" if float(m["damping"]) != REFERENCE["damping"] else "",
             f"ceiling {m['clamp']}" if float(m["clamp"]) != REFERENCE["clamp"] else "",
             _channels(int(m["channels"])) if m["channels"] else "", *extra]
    parts = [p for p in parts if p]
    return name + (f" ({', '.join(parts)})" if parts else "")


def read(name: str) -> str:
    """A read's qualifier: '' for the primary read, else what differs from it."""
    out = ["whole clip"] if "@wholeclip" in name else []
    return ", ".join(out + (["with rotation rates"] if "+rate" in name else []))

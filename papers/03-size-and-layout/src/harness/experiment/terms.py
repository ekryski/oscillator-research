"""The paper's terms for the record's labels, in one place.

The record keeps paper 02's labels (`floor`, `field-...`, `severed-...`,
`bank-c4`, `ann-gru`, `envelope`), because they are the runs' identities and
because paper 03 takes cells from paper 02's record under the same ones.
Everything people read (tables, figures, the summary report) names things
with the terms the paper's glossary defines (Appendix A), and gets them from
here. The same mapping is tabulated in src/README.md.

Paper 03's labels add four suffixes to paper 02's: `-c<C>` for a channel
count other than 4, `-<G>x<G>` for a lattice other than 16 x 16, `-16bands`
when paper 02's 16 mel bands are mapped onto the lattice's rows rather than
one band driving each row, and `-w<N>` for an analysis window of N samples
other than paper 02's 512. A trained baseline carries the suffixes of the
network it is sized to.
"""

from __future__ import annotations

import re

#: arms named by kind
ARMS = {
    "floor": "spectrogram-only baseline",
    "ann-gru": "GRU",
    "ann-tcn": "TCN",
    "ann-cnn": "CNN",
    "ann-transformer": "transformer",
    "ann-s4d": "S4D",
}

#: the design factors, as the Arm fields name them
FACTORS = {"physics": "coupling function", "boundary": "lattice geometry", "omega": "natural frequencies",
           "damping": "restoring strength", "clamp": "coupling ceiling", "channels": "channels",
           "grid": "lattice", "bands": "band mapping"}

#: a factor's levels
LEVELS = {
    "kuramoto": "Kuramoto", "sakaguchi": "Kuramoto–Sakaguchi", "harmonic2": "second harmonic",
    "winfree": "Winfree", "sl": "Stuart–Landau", "sl-fixedamp": "Stuart–Landau, fixed amplitude",
    "random": "random", "designed": "tonotopic", "uniform": "identical",
}

#: the input pathways, as the record names them
PATHWAYS = {"envelope": "band-energy", "quadrature": "quadrature", "carrier": "carrier"}

#: paper 02's reference configuration of the coupled network, and its size
REFERENCE = {"physics": "kuramoto", "boundary": "torus", "omega": "random", "damping": 0.3, "clamp": 1.0}
CHANNELS, GRID = 4, 16

_SUFFIX = re.compile(r"^(?P<base>.*?)(?:-c(?P<channels>\d+))?(?:-(?P<grid>\d+)x\d+)?(?:-(?P<bands>\d+)bands)?"
                     r"(?:-w(?P<window>\d+))?$")
_OSCILLATORS = re.compile(r"^(field|severed)-(?P<physics>sl-fixedamp|[a-z0-9]+)-(?P<boundary>[a-z]+)-"
                          r"(?P<omega>[a-z]+)-lam(?P<damping>[0-9.]+)-clamp(?P<clamp>[0-9.]+)$")


def level(value) -> str:
    """A factor level in the paper's words: 'designed' -> 'tonotopic', 0.3 -> '0.3'."""
    return LEVELS.get(value, value) if isinstance(value, str) else f"{value:g}"


def channels_text(n: int) -> str:
    return f"{n} channel" + ("" if n == 1 else "s")


def lattice_text(grid: int, bands: int | None) -> str:
    """'32 × 32 lattice, 32 mel bands' or '32 × 32 lattice, 16 mel bands mapped onto 32 rows'."""
    if bands:
        return f"{grid} × {grid} lattice, {bands} mel bands mapped onto {grid} rows"
    return f"{grid} × {grid} lattice, {grid} mel bands"


def rows_text(grid: int, bands: int | None) -> str:
    """What the spectrogram-only baseline reads: '32 mel bands' or '16 mel bands mapped onto 32 rows'."""
    return f"{bands} mel bands mapped onto {grid} rows" if bands else f"{grid} mel bands"


def split(label: str) -> tuple[str, int, int, int | None]:
    """(base label, channels, grid, mapped bands or None) of a record label."""
    m = _SUFFIX.match(label)
    base = m["base"]
    if m["window"]:
        base = base.removesuffix(f"-w{m['window']}")
    if base.startswith("bank") and m["channels"]:
        return "bank", int(m["channels"]), int(m["grid"] or GRID), int(m["bands"]) if m["bands"] else None
    return (base, int(m["channels"] or CHANNELS), int(m["grid"] or GRID),
            int(m["bands"]) if m["bands"] else None)


def _size(channels: int, grid: int, bands: int | None, always: bool = False) -> list[str]:
    """What a label says about its size, in words; nothing for paper 02's size unless `always`."""
    parts = []
    if always or channels != CHANNELS or grid != GRID or bands:
        parts.append(channels_text(channels))
        parts.append(lattice_text(grid, bands) if (grid != GRID or bands) else f"{GRID} × {GRID} lattice")
    return parts


def window_of(label: str) -> int | None:
    """The analysis window a label names, in samples, or None for paper 02's 512."""
    m = _SUFFIX.match(label)
    return int(m["window"]) if m["window"] else None


def arm(label: str, tier: str | None = None) -> str:
    """The paper's name for a record label, with its analysis window where it is not paper 02's."""
    name = _arm(label)
    window = window_of(label)
    if window is None:
        return name
    text = f"{window:,}-sample window"
    return name[:-1] + f", {text})" if name.endswith(")") else f"{name} ({text})"


def _arm(label: str) -> str:
    """The paper's name for a record label.

    Paper 02's reference network at paper 02's size is named plainly; any
    other configuration names the factors where it departs from it. Every
    leaky-integrator bank in paper 03 is state-matched to the network with
    its own channel count and lattice (paper 02's `bank-c8`, its width-matched
    bank, is paper 03's state-matched bank at 8 channels). A trained baseline
    is named with the network it is sized to.
    """
    base, channels, grid, bands = split(label)
    if base == "bank":
        return f"leaky-integrator bank, state-matched ({', '.join(_size(channels, grid, bands, always=True))})"
    if base == "floor":
        return ARMS[base] + (f" ({rows_text(grid, bands)})" if (grid != GRID or bands) else "")
    if base in ARMS:
        extra = _size(channels, grid, bands)
        return ARMS[base] + (f" (sized to {', '.join(extra)})" if extra else "")
    m = _OSCILLATORS.match(base)
    if not m:
        return label
    name = "uncoupled oscillator network" if label.startswith("severed") else "coupled oscillator network"
    parts = [level(m["physics"]) if m["physics"] != REFERENCE["physics"] else "",
             m["boundary"] if m["boundary"] != REFERENCE["boundary"] else "",
             f"{level(m['omega'])} ω" if m["omega"] != REFERENCE["omega"] else "",
             f"λ {m['damping']}" if float(m["damping"]) != REFERENCE["damping"] else "",
             f"ceiling {m['clamp']}" if float(m["clamp"]) != REFERENCE["clamp"] else "",
             *_size(channels, grid, bands)]
    parts = [p for p in parts if p]
    return name + (f" ({', '.join(parts)})" if parts else "")


def read(name: str) -> str:
    """A read's qualifier: '' for the primary read, else what differs from it."""
    out = ["whole clip"] if "@wholeclip" in name else []
    return ", ".join(out + (["with rotation rates"] if "+rate" in name else []))

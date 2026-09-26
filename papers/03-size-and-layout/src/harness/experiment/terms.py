"""The paper's terms for the record's labels, in one place.

The record's labels follow the paper's glossary (Appendix A) in short form,
exactly as paper 02's do: `baseline`,
`coupled-kuramoto-torus-random-restoring0.3-ceiling1`, `uncoupled-...`,
`bank-state`, `bank-width`, `trained-gru`, the `spectrogram` pathway. Paper 03
takes cells from paper 02's record under the same labels. Everything people
read (tables, figures, the summary report) names things in full with the words
defined here, so no two places can drift apart. The same mapping is tabulated
in src/README.md.

Paper 03's labels extend paper 02's with its size, in this order and only
where it is not paper 02's: `-ch<C>` for a channel count other than 4 (a bank
names its channels in its base label: `bank-state` is 4 channels, `bank-width`
8, `bank-ch<C>` any other), `-<G>x<G>` for a lattice other than 16 x 16,
`-16bands` when paper 02's 16 mel bands are mapped onto the lattice's rows
rather than one band driving each row, and `-w<N>` for an analysis window of N
samples other than paper 02's 512. A trained baseline carries the suffixes of
the network it is sized to. So `coupled-kuramoto-torus-random-restoring0.3-
ceiling1-ch16-64x64-w1024` is the reference network at 16 channels of a
64 x 64 lattice, one band per row, under a 1,024-sample window.
"""

from __future__ import annotations

import re

#: arms named by base label
ARMS = {
    "baseline": "spectrogram-only baseline",
    "trained-gru": "GRU",
    "trained-tcn": "TCN",
    "trained-cnn": "CNN",
    "trained-transformer": "transformer",
    "trained-s4d": "S4D",
}

#: the design factors, as the Arm fields name them
FACTORS = {"coupling": "coupling function", "geometry": "lattice geometry", "frequencies": "natural frequencies",
           "restoring": "restoring strength", "ceiling": "coupling ceiling", "channels": "channels",
           "grid": "lattice", "bands": "band mapping", "window": "analysis window"}

#: a factor's levels
LEVELS = {
    "kuramoto": "Kuramoto", "kuramoto-sakaguchi": "Kuramoto–Sakaguchi", "second-harmonic": "second harmonic",
    "winfree": "Winfree", "stuart-landau": "Stuart–Landau", "stuart-landau-fixed": "Stuart–Landau, fixed amplitude",
    "random": "random", "tonotopic": "tonotopic", "identical": "identical",
}

#: the input pathways
PATHWAYS = {"spectrogram": "spectrogram", "quadrature": "quadrature"}

#: paper 02's reference configuration of the coupled network, and its size
REFERENCE = {"coupling": "kuramoto", "geometry": "torus", "frequencies": "random", "restoring": 0.3, "ceiling": 1.0}
CHANNELS, GRID = 4, 16
#: a bank's channel count, by its base label
BANK_CHANNELS = {"bank-state": 4, "bank-width": 8}

_SUFFIX = re.compile(r"^(?P<base>.*?)(?:-ch(?P<channels>\d+))?(?:-(?P<grid>\d+)x\d+)?(?:-(?P<bands>\d+)bands)?"
                     r"(?:-w(?P<window>\d+))?$")
_NETWORK = re.compile(r"^(coupled|uncoupled)-"
                      r"(?P<coupling>kuramoto-sakaguchi|second-harmonic|stuart-landau-fixed|stuart-landau|[a-z]+)-"
                      r"(?P<geometry>[a-z]+)-(?P<frequencies>[a-z]+)-restoring(?P<restoring>[0-9.]+)"
                      r"-ceiling(?P<ceiling>[0-9.]+)$")


def level(value) -> str:
    """A factor level in the paper's words: 'second-harmonic' -> 'second harmonic', 0.3 -> '0.3'."""
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
    """(base label, channels, grid, mapped bands or None) of a record label. A bank's base is "bank"."""
    m = _SUFFIX.match(label)
    base, channels = m["base"], int(m["channels"]) if m["channels"] else CHANNELS
    if base in BANK_CHANNELS:                    # bank-state, bank-width; bank-ch<C> parses as it stands
        base, channels = "bank", BANK_CHANNELS[base]
    return base, channels, int(m["grid"] or GRID), int(m["bands"]) if m["bands"] else None


def window_of(label: str) -> int | None:
    """The analysis window a label names, in samples, or None for paper 02's 512."""
    m = _SUFFIX.match(label)
    return int(m["window"]) if m["window"] else None


def _size(channels: int, grid: int, bands: int | None, always: bool = False) -> list[str]:
    """What a label says about its size, in words; nothing for paper 02's size unless `always`."""
    parts = []
    if always or channels != CHANNELS or grid != GRID or bands:
        parts.append(channels_text(channels))
        parts.append(lattice_text(grid, bands) if (grid != GRID or bands) else f"{GRID} × {GRID} lattice")
    return parts


def arm(label: str) -> str:
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
    its own channel count and lattice (`bank-width`, paper 02's width-matched
    bank of the 4-channel network, is the state-matched bank of 8 channels). A
    trained baseline is named with the network it is sized to.
    """
    base, channels, grid, bands = split(label)
    if base == "bank":
        return f"leaky-integrator bank, state-matched ({', '.join(_size(channels, grid, bands, always=True))})"
    if base == "baseline":
        return ARMS[base] + (f" ({rows_text(grid, bands)})" if (grid != GRID or bands) else "")
    if base in ARMS:
        extra = _size(channels, grid, bands)
        return ARMS[base] + (f" (sized to {', '.join(extra)})" if extra else "")
    m = _NETWORK.match(base)
    if not m:
        return label
    name = "uncoupled oscillator network" if label.startswith("uncoupled") else "coupled oscillator network"
    parts = [level(m["coupling"]) if m["coupling"] != REFERENCE["coupling"] else "",
             m["geometry"] if m["geometry"] != REFERENCE["geometry"] else "",
             f"{level(m['frequencies'])} ω" if m["frequencies"] != REFERENCE["frequencies"] else "",
             f"λ {m['restoring']}" if float(m["restoring"]) != REFERENCE["restoring"] else "",
             f"ceiling {m['ceiling']}" if float(m["ceiling"]) != REFERENCE["ceiling"] else "",
             *_size(channels, grid, bands)]
    parts = [p for p in parts if p]
    return name + (f" ({', '.join(parts)})" if parts else "")


def read(name: str) -> str:
    """A read's qualifier: '' for the primary read, else what differs from it."""
    out = ["whole clip"] if "@wholeclip" in name else []
    return ", ".join(out + (["with rotation rates"] if "+rate" in name else []))

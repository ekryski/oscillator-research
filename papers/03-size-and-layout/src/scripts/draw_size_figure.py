"""Draw the schematic of what paper 03 varies: lattice size, channel count and band mapping.

    uv run python scripts/draw_size_figure.py

A drawing, not a result: nothing here reads the record. The numbers in it come
from the plan (harness.experiment.plan) and the arms (harness.experiment.arms), so
the figure cannot drift from the design. Writes
resources/figures/a4-size-and-band-mapping.{pdf,png}.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Rectangle

from harness.experiment import plan
from harness.experiment import stream as st
from harness.experiment.arms import Arm
from harness.utils.paths import FIGURES_DIR

INK, MUTED, NETWORK = "#222222", "#8C8C8C", "#534AB7"
SIZE = colormaps["Blues"]
BAND = colormaps["viridis"]


def size_panel(ax) -> None:
    """Rows: lattice side G; columns: channel count C; each cell the number of oscillators C * G * G."""
    ax.set_title("(a) Lattice size × channel count: the number of oscillators", fontsize=10, loc="left")
    grids, channels = plan.GRIDS, plan.CHANNELS
    counts = [Arm("field", channels=c, grid=g).states for g in grids for c in channels]
    lo, hi = math.log2(min(counts)), math.log2(max(counts))
    for i, g in enumerate(grids):
        y = len(grids) - 1 - i
        ax.text(-0.15, y + 0.5, f"{g} × {g}", ha="right", va="center", fontsize=8)
        for j, c in enumerate(channels):
            arm = Arm("field", channels=c, grid=g)
            shade = 0.15 + 0.7 * (math.log2(arm.states) - lo) / (hi - lo)
            ax.add_patch(Rectangle((j, y), 1, 1, fc=SIZE(shade), ec="white", lw=1.5))
            color = "white" if shade > 0.55 else INK
            ax.text(j + 0.5, y + 0.58, f"{arm.states:,}", ha="center", va="center", fontsize=7.5, color=color)
            ax.text(j + 0.5, y + 0.28, f"{arm.budget:,} p", ha="center", va="center", fontsize=5.5, color=color)
            if arm.states > st.STREAM_STATES:
                ax.add_patch(Rectangle((j + 0.06, y + 0.06), 0.88, 0.88, fill=False, ec=color, lw=0.6, ls=":"))
    # paper 02's network, drawn last so no neighbouring cell's edge covers it
    j, y = channels.index(4), len(grids) - 1 - grids.index(16)
    ax.add_patch(Rectangle((j + 0.03, y + 0.03), 0.94, 0.94, fill=False, ec=NETWORK, lw=2.2, zorder=5))
    for j, c in enumerate(channels):
        ax.text(j + 0.5, len(grids) + 0.12, f"{c}", ha="center", va="bottom", fontsize=8)
    ax.text(len(channels) / 2, len(grids) + 0.55, "channels", ha="center", va="bottom", fontsize=8)
    ax.text(-0.15, len(grids) + 0.12, "lattice", ha="right", va="bottom", fontsize=8)
    ax.text(0, -0.25, "Each cell: oscillators (C · G²) and parameters (2 · C · G²). Equal shades hold equally many\n"
            "oscillators in different layouts. Purple: paper 02's network. Dotted: more than 4,096 states,\n"
            "read channel by channel.", ha="left", va="top", fontsize=7.5, color=MUTED)
    ax.set_xlim(-1.3, len(channels) + 0.1)
    ax.set_ylim(-1.6, len(grids) + 1.1)


def mapping(ax, x0: float, bands: int, rows: int, title: str) -> None:
    """Bands on the left, lattice rows on the right: each row coloured by the band or bands that drive
    it, and joined to them (harness.experiment.protocol.to_rows: one band per row, a band on rows / bands
    adjacent rows, or the mean of bands / rows adjacent bands)."""
    h, w, gap = 5.0, 0.35, 1.0
    ax.text(x0 + (2 * w + gap) / 2, h + 0.35, title, ha="center", va="bottom", fontsize=8)
    xr = x0 + w + gap
    for b in range(bands):
        y0, y1 = h * b / bands, h * (b + 1) / bands
        ax.add_patch(Rectangle((x0, y0), w, y1 - y0, fc=BAND(b / (bands - 1)), ec="white", lw=0.3))
    for r in range(rows):
        y0, y1 = h * r / rows, h * (r + 1) / rows
        drivers = ([r * bands // rows] if rows >= bands
                   else list(range(r * bands // rows, (r + 1) * bands // rows)))
        for k, b in enumerate(drivers):                        # a row driven by two bands shows both
            part = w / len(drivers)
            ax.add_patch(Rectangle((xr + k * part, y0), part, y1 - y0, fc=BAND(b / (bands - 1)), lw=0))
            yb = h * (b + 0.5) / bands
            ax.plot([x0 + w, xr], [yb, (y0 + y1) / 2], color=BAND(b / (bands - 1)), lw=0.6, alpha=0.8)
        ax.add_patch(Rectangle((xr, y0), w, y1 - y0, fill=False, ec="white", lw=0.3))
    ax.add_patch(Rectangle((xr, 0), w, h, fill=False, ec=NETWORK, lw=0.9))
    ax.text(x0 + w / 2, -0.2, f"{bands}\nmel bands", ha="center", va="top", fontsize=7)
    ax.text(xr + w / 2, -0.2, f"{rows}\nrows", ha="center", va="top", fontsize=7)


def mapping_panel(ax) -> None:
    ax.set_title("(b) Band mapping: how the mel bands drive a lattice's rows", fontsize=10, loc="left")
    mapping(ax, 0.0, 32, 32, "one band per row\n(32 × 32)")
    mapping(ax, 2.9, 16, 32, "16 bands mapped onto\n32 rows (32 × 32)")
    mapping(ax, 5.8, 16, 8, "16 bands mapped onto\n8 rows (8 × 8)")
    ax.text(0, -1.25, "A lattice of G rows is driven by G mel bands, one per row, or by paper 02's 16 bands mapped\n"
            "onto its rows, in frequency order (lowest at the bottom); at 16 × 16 the two are the same.\n"
            "Every row of every channel is driven alike; nothing in the mapping is fitted.",
            ha="left", va="top", fontsize=7.5, color=MUTED)
    ax.set_xlim(-0.3, 7.9)
    ax.set_ylim(-2.4, 6.2)


def main() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.05, 1]})
    size_panel(a)
    mapping_panel(b)
    for ax in (a, b):
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    stem = FIGURES_DIR / "a4-size-and-band-mapping"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 200})):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    print(f"wrote {stem}.{{pdf,png}}")


if __name__ == "__main__":
    main()

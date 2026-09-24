"""Draw the appendix schematic: layers stack in depth, channels sit side by side.

    cd src && uv run python ../scripts/draw_channels_figure.py      # from the paper's folder

A drawing, not a result: nothing here reads the record. Writes
resources/figures/a1-channels-and-layers.{pdf,png}.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import FancyArrowPatch, Rectangle

from harness.utils.paths import FIGURES_DIR

BANDS = 16
INK, MUTED, NETWORK = "#222222", "#8C8C8C", "#534AB7"
BAND = colormaps["viridis"]


def arrow(ax, start, end, color=INK, style="-|>", lw=1.1):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=10, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def spectrogram(ax, x, y, w, h, label=True):
    """The mel spectrogram: 16 band rows, lowest at the bottom, coloured by band."""
    for r in range(BANDS):
        ax.add_patch(Rectangle((x, y + r * h / BANDS), w, h / BANDS, color=BAND(r / (BANDS - 1)), lw=0))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=INK, lw=0.8))
    if label:
        ax.text(x + w / 2, y - 0.25, "mel spectrogram\n16 band energies", ha="center", va="top", fontsize=8)


def lattice(ax, x, y, s, name):
    """One channel: a 16 x 16 lattice whose row r is driven by band r, shown by its colour."""
    cell = s / BANDS
    for r in range(BANDS):
        ax.add_patch(Rectangle((x, y + r * cell), s, cell, color=BAND(r / (BANDS - 1)), alpha=0.35, lw=0))
    for i in range(BANDS + 1):
        ax.plot([x, x + s], [y + i * cell] * 2, color="white", lw=0.35)
        ax.plot([x + i * cell] * 2, [y, y + s], color="white", lw=0.35)
    ax.add_patch(Rectangle((x, y), s, s, fill=False, ec=NETWORK, lw=1.2))
    ax.text(x + s / 2, y + s + 0.12, name, ha="center", va="bottom", fontsize=8, color=NETWORK)


def box(ax, x, y, w, h, text, color=INK, fill="white"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=color, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=color)


def layers_panel(ax):
    ax.set_title("(a) Layers: stacked, one after another (depth)", fontsize=10, loc="left")
    spectrogram(ax, 0.2, 2.0, 0.9, 2.0)
    xs = [1.9, 3.5, 5.1]
    for i, x in enumerate(xs):
        box(ax, x, 2.2, 1.1, 1.6, f"layer {i + 1}", color=MUTED)
    arrow(ax, (1.1, 3.0), (xs[0], 3.0))
    for a, b in zip(xs, xs[1:], strict=False):
        arrow(ax, (a + 1.1, 3.0), (b, 3.0))
    box(ax, 6.7, 2.6, 1.0, 0.8, "readout")
    arrow(ax, (xs[-1] + 1.1, 3.0), (6.7, 3.0))
    ax.text(4.0, 0.6, "each layer transforms the previous layer's output;\n"
            "a deep network or a transformer stacks many", ha="center", va="top", fontsize=8, color=MUTED)


def channels_panel(ax):
    ax.set_title("(b) Channels: side by side, all driven by the same input (width)", fontsize=10, loc="left")
    spectrogram(ax, 0.2, 2.0, 0.9, 2.0)
    s, x0 = 1.25, 2.6
    ys = [4.75, 3.05, 1.35, -0.35]
    for i, y in enumerate(ys):
        lattice(ax, x0, y, s, f"channel {i + 1}: 16 × 16 lattice")
        arrow(ax, (1.1, 3.0), (x0 - 0.05, y + s / 2), color=INK)
        arrow(ax, (x0 + s + 0.05, y + s / 2), (6.7, 3.0), color=NETWORK)
    box(ax, 6.7, 2.6, 1.0, 0.8, "readout")
    ax.text(6.0, 5.6, "no coupling between channels", ha="center", va="center", fontsize=8, color=MUTED)
    ax.text(0.65, 0.95, "band r drives\nrow r of\nevery channel", ha="center", va="top", fontsize=8, color=MUTED)
    ax.text(4.2, -0.75, "each channel has its own coupling kernel and natural frequencies; within a channel every\n"
            "oscillator is coupled to every other. The readout reads all channels' signals together,\n"
            "as the attention heads of one transformer layer are read together.",
            ha="center", va="top", fontsize=8, color=MUTED)


def main() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1]})
    layers_panel(a)
    channels_panel(b)
    for ax in (a, b):
        ax.set_xlim(0, 8)
        ax.set_ylim(-1.9, 6.4)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    stem = FIGURES_DIR / "a1-channels-and-layers"
    for suffix, kwargs in ((".pdf", {"metadata": {"CreationDate": None, "Creator": None, "Producer": None}}),
                           (".png", {"dpi": 200, "metadata": {"Software": None}})):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    print(f"wrote {stem}.{{pdf,png}}")


if __name__ == "__main__":
    main()

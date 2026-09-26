"""Draw the appendix schematic: how each kind of arm is read, from the spectrogram to the digit.

    cd src && uv run python ../scripts/draw_read_figure.py      # from the paper's folder

A drawing, not a result: nothing here reads the record. Writes
resources/figures/fig10-appD1-readout.{pdf,png}.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from harness.utils.paths import FIGURES_DIR

BANDS = 16
INK, MUTED = "#222222", "#6E6E6E"
INPUT, NETWORK, TRAINED = "#6E6E6E", "#534AB7", "#0F6E56"
BAND = colormaps["viridis"]
SAVE = ((".pdf", {"metadata": {"CreationDate": None, "Creator": None, "Producer": None}}),
        (".png", {"dpi": 200, "metadata": {"Software": None}}))

#: the rows: (name, colour, what produces the signals, the statistics' width, what brings them to 192)
ROWS = (
    ("Spectrogram-only baseline", INPUT, "no reservoir:\nthe 16 band energies\nthemselves",
     "192 features", "standardize\n(already 192 wide)"),
    ("Reservoir, untrained: an oscillator network or a leaky-integrator bank", NETWORK,
     "e.g. 1,024 oscillators,\nsine and cosine of each\nphase: 2,048 signals", "12,288 to 24,576 features",
     "standardize, then a\nrandom projection to 192"),
    ("Trained baseline: GRU, TCN, CNN, transformer or S4D", TRAINED, "a small network trained\nend to end: 16 hidden\nunits (GRU: 18)",
     "192 features\n(GRU: 216)", "standardize\n(GRU: projected to 192)"),
)


def arrow(ax, start, end, color=INK, style="-|>", lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=10, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def box(ax, x, y, w, h, text, color=INK, fontsize=7.5, lw=1.1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white",
                                ec=color, lw=lw, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color)


def spectrogram(ax, x, y, w, h):
    for r in range(BANDS):
        ax.add_patch(Rectangle((x, y + r * h / BANDS), w, h / BANDS, color=BAND(r / (BANDS - 1)), lw=0))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=INK, lw=0.8))
    ax.text(x + w / 2, y - 0.2, "mel spectrogram:\n16 band energies\nper frame", ha="center", va="top", fontsize=7.5)


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(-0.3, 6.6)
    ax.axis("off")
    spectrogram(ax, 0.15, 2.3, 0.75, 2.2)

    col_src, col_stats, col_fit = (1.9, 2.5), (4.95, 3.1), (8.6, 2.4)    # (x, width) of each column
    readout = (11.65, 0.9, 1.85, 4.8)
    centres = (5.3, 3.25, 1.2)
    h = 1.05
    for (name, colour, source, stats, fit), yc in zip(ROWS, centres, strict=True):
        y = yc - h / 2
        ax.text(col_src[0], y + h + 0.1, name, fontsize=8.5, color=colour, fontweight="bold", va="bottom")
        arrow(ax, (0.9, 3.4), (col_src[0], yc), color=MUTED, lw=0.8)
        box(ax, col_src[0], y, col_src[1], h, source, colour)
        arrow(ax, (col_src[0] + col_src[1], yc), (col_stats[0], yc))
        box(ax, col_stats[0], y, col_stats[1], h,
            "mean, standard deviation and\nmean absolute change of each\nsignal over 4 windows:\n" + stats, INK)
        arrow(ax, (col_stats[0] + col_stats[1], yc), (col_fit[0], yc))
        box(ax, col_fit[0], y, col_fit[1], h, fit, INK)
        arrow(ax, (col_fit[0] + col_fit[1], yc), (readout[0], yc))
    # the trained baselines learn through their own head on the same statistics, used only in training
    y3 = centres[2] - h / 2
    box(ax, col_stats[0] + 0.25, -0.25, col_stats[1] - 0.5, 0.5, "learned linear head\n(training only)", TRAINED,
        fontsize=7, ls="--")
    arrow(ax, (col_stats[0] + col_stats[1] / 2, y3), (col_stats[0] + col_stats[1] / 2, 0.25), color=TRAINED,
          lw=0.9, ls="--")
    arrow(ax, (col_stats[0] + 0.25, 0.0), (col_src[0] + col_src[1] / 2, y3), color=TRAINED, lw=0.9, ls="--")
    ax.text(col_src[0] + 0.05, 0.1, "gradients train\nthe network", fontsize=6.5, color=TRAINED, va="center")
    # one readout form for every arm, fitted separately to each
    box(ax, readout[0], readout[1], readout[2], readout[3],
        "ridge readout:\none linear layer,\n192 features\n→ 10 digit scores,\nfitted in closed\nform, the same\nfor every arm", INK,
        fontsize=8, lw=1.4)
    arrow(ax, (readout[0] + readout[2], readout[1] + readout[3] / 2), (14.3, readout[1] + readout[3] / 2))
    ax.text(14.4, readout[1] + readout[3] / 2, "digit", fontsize=9, va="center")
    ax.text(0.1, 6.5, "Order task: one window over frames 16 to 147, so the baseline and the trained baselines have "
            "48 to 54 features and are read as they are; the reservoirs' 3,072 to 6,144 are projected to 192.",
            fontsize=7.5, color=MUTED, va="top")
    fig.tight_layout()
    path = FIGURES_DIR / "fig10-appD1-readout"
    for suffix, kwargs in SAVE:
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {path}.{{pdf,png}}")


if __name__ == "__main__":
    main()

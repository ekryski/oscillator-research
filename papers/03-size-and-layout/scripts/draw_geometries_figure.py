"""Draw the appendix schematic of the eight lattice geometries.

    cd src && uv run python ../scripts/draw_geometries_figure.py      # from the paper's folder

A drawing, not a result: nothing here reads the record. Copied from paper
02's script, and drawn as paper 02 draws it, at 16 x 16: at any other lattice
size only the number of rows and columns changes, the cube's slab, and the
coil's sites to a turn (G / 4 rows). Every channel stores its 256 oscillators
the same way, as a 16 x 16 grid whose row r is driven by mel band r. A geometry changes only which oscillators are neighbours, which is
how the grid's edges are glued. Each geometry is drawn twice: the flat grid,
with its glued edges and one oscillator's nearest neighbours, and the shape the
gluing makes, with the same oscillator and neighbours. The neighbour rules
follow harness/models/geometries/. Writes
resources/figures/a2-lattice-geometries.{pdf,png}.
"""

from __future__ import annotations

import math
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from harness.utils.paths import FIGURES_DIR

G = 16
BAND = colormaps["viridis"]
INK, MUTED, GLUE_A, GLUE_B, STAR = "#222222", "#8C8C8C", "#D85A30", "#0F6E56", "#E4002B"
#: the oscillator whose neighbours are shown: the highest band's row, the first column
ORIGIN = (15, 0)


# ---------------------------------------------------------------------------
# Neighbours of ORIGIN on the flat grid, (row, column), per geometry
# ---------------------------------------------------------------------------

def _grid_neighbours(wrap_rows: bool, wrap_cols: bool) -> list[tuple[int, int]]:
    r, c = ORIGIN
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if not 0 <= rr < G:
            if not wrap_rows:
                continue
            rr %= G
        if not 0 <= cc < G:
            if not wrap_cols:
                continue
            cc %= G
        out.append((rr, cc))
    return out


def _helix_neighbours() -> list[tuple[int, int]]:
    """Ring distance 1, and one turn (64 sites, an octave) either way, on the closed ring."""
    p = ORIGIN[0] * G + ORIGIN[1]
    return [divmod((p + d) % (G * G), G) for d in (1, -1, 64, -64)]


def _cube_neighbours() -> list[tuple[int, int]]:
    """Column c is (c // 4, c % 4) in its row's 4 x 4 slab; all three axes wrap."""
    r, c = ORIGIN
    a, b = divmod(c, 4)
    out = [((r + 1) % G, c), ((r - 1) % G, c)]
    out += [(r, ((a + d) % 4) * 4 + b) for d in (1, -1)]
    out += [(r, a * 4 + (b + d) % 4) for d in (1, -1)]
    return out


# ---------------------------------------------------------------------------
# Where a (row, column) site sits on each shape
# ---------------------------------------------------------------------------

def on_sheet(r, c):
    return c + 0.5, 0.0, r + 0.5


def on_cylinder(r, c):
    t = 2 * math.pi * (c + 0.5) / G
    return 4 * math.cos(t), 4 * math.sin(t), r + 0.5


def on_torus(r, c):
    u, v = 2 * math.pi * (r + 0.5) / G, 2 * math.pi * (c + 0.5) / G
    return (6 + 2.2 * math.cos(v)) * math.cos(u), (6 + 2.2 * math.cos(v)) * math.sin(u), 2.2 * math.sin(v)


def on_sphere(r, c):
    lat = -math.pi / 2 + math.pi * (r + 0.5) / G
    lon = 2 * math.pi * (c + 0.5) / G
    return 5 * math.cos(lat) * math.cos(lon), 5 * math.cos(lat) * math.sin(lon), 5 * math.sin(lat)


def on_helix(r, c):
    p = r * G + c
    t = 2 * math.pi * p / 64
    return 4 * math.cos(t), 4 * math.sin(t), p / 64 * 3.2


#: the coil's radius at the apex, as a share of the base's (the cochlea's curvature weighting)
APEX_RADIUS = 0.25


def _coil_neighbours() -> list[tuple[int, int]]:
    """Coil distance 1, and one turn either way, on the open coil: past either end there is nothing."""
    p = ORIGIN[0] * G + ORIGIN[1]
    return [divmod(q, G) for q in (p + 1, p - 1, p + 64, p - 64) if 0 <= q < G * G]


def on_coil(r, c):
    """A snail shell: the apex (lowest band, p = 0) at the centre and top, the base on the outside."""
    p = r * G + c
    t = 2 * math.pi * p / 64
    radius = 4 * APEX_RADIUS ** (1 - p / (G * G - 1))
    return radius * math.cos(t), radius * math.sin(t), (1 - p / (G * G - 1)) * 2.5


def on_cube(r, c):
    a, b = divmod(c, 4)
    return a + 0.5, b + 0.5, r * 0.55 + 0.25


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def flat(ax, neighbours, glue_rows, glue_cols, note, cube=False):
    img = np.repeat(np.arange(G)[:, None], G, axis=1)
    ax.imshow(img, origin="lower", cmap=BAND, alpha=0.45, extent=(0, G, 0, G), vmin=0, vmax=G - 1)
    for i in range(G + 1):
        width = 1.2 if cube and i % 4 == 0 else 0.3
        ax.plot([0, G], [i, i], color="white", lw=0.3)
        ax.plot([i, i], [0, G], color="white", lw=width)
    # edges: glued pairs share a colour and an arrow; open edges are plain and dark; None is neither
    # (the helix's rows run on into each other, the cube's columns wrap within each slab)
    for side, glued, colour in (("cols", glue_cols, GLUE_A), ("rows", glue_rows, GLUE_B)):
        if side == "cols":
            ends = (((0, 0), (0, G)), ((G, 0), (G, G)))
        else:
            ends = (((0, 0), (G, 0)), ((0, G), (G, G)))
        for (x0, y0), (x1, y1) in ends:
            if glued is None:
                ax.plot([x0, x1], [y0, y1], color=MUTED, lw=0.8)
            elif glued:
                ax.plot([x0, x1], [y0, y1], color=colour, lw=2.4, solid_capstyle="butt")
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                dx, dy = (x1 - x0) * 0.08, (y1 - y0) * 0.08
                ax.annotate("", xy=(mx + dx, my + dy), xytext=(mx - dx, my - dy),
                            arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 2.0})
            else:
                ax.plot([x0, x1], [y0, y1], color=INK, lw=2.4)
    ys, xs = zip(*neighbours, strict=True)
    ax.scatter(np.array(xs) + 0.5, np.array(ys) + 0.5, s=22, color="white", edgecolor=INK, zorder=5, lw=0.8)
    ax.scatter([ORIGIN[1] + 0.5], [ORIGIN[0] + 0.5], marker="*", s=90, color=STAR, zorder=6)
    ax.text(G / 2, -1.2, note, ha="center", va="top", fontsize=7, color=MUTED)
    ax.set_xlim(-0.8, G + 0.8)
    ax.set_ylim(-0.8, G + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")


def surface(ax, place, rows=G, cols=G + 1):
    """A surface through every site, coloured by band."""
    r = np.linspace(0, G, 60)
    c = np.linspace(0, G, 60)
    R, C = np.meshgrid(r, c, indexing="ij")
    pts = np.vectorize(lambda a, b: place(a - 0.5, b - 0.5), otypes=[float, float, float])(R, C)
    colours = BAND(np.clip(R / G, 0, 1))
    ax.plot_surface(*pts, facecolors=colours, alpha=0.8, linewidth=0, shade=False, antialiased=True)


def mark(ax, place, neighbours):
    for r, c in neighbours:
        ax.scatter(*place(r, c), s=22, color="white", edgecolor=INK, depthshade=False, zorder=5, lw=0.8)
    ax.scatter(*place(*ORIGIN), marker="*", s=110, color=STAR, depthshade=False, zorder=6)


def shape_axes(ax, view=(22, -60)):
    ax.computed_zorder = False           # the marked oscillators stay in front of the surface
    ax.view_init(*view)
    ax.set_axis_off()
    ax.set_box_aspect(None, zoom=1.15)


def draw_helix(ax, neighbours):
    p = np.arange(G * G + 1)
    t = 2 * np.pi * p / 64
    x, y, z = 4 * np.cos(t), 4 * np.sin(t), p / 64 * 3.2
    for i in range(G * G):
        ax.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2], color=BAND((i // G) / (G - 1)), lw=2.2)
    ax.plot([x[-1], x[0]], [y[-1], y[0]], [z[-1], z[0]], color=MUTED, lw=1, ls="--")
    mark(ax, on_helix, neighbours)


def draw_coil(ax, neighbours, cochlea: bool = False):
    """The open coil, coloured by band; the cochlea's line thickens with curvature toward the apex, and its
    arrows point the way influence runs most strongly, from base to apex."""
    n = G * G
    pts = np.array([on_coil(*divmod(q, G)) for q in range(n)])
    for i in range(n - 1):
        w = APEX_RADIUS ** (i / (n - 1)) if cochlea else 1.0          # 1 at the apex, a quarter at the base
        ax.plot(*pts[i:i + 2].T, color=BAND((i // G) / (G - 1)), lw=1.0 + 2.6 * w if cochlea else 2.2)
    if cochlea:
        for q in (228, 196, 160, 128):                           # on the outer turns, where they show
            a, b = pts[q], pts[q - 5]
            ax.quiver(*a, *(b - a), color=INK, arrow_length_ratio=0.45, lw=1.3)
    mark(ax, on_coil, neighbours)


def draw_cube(ax, neighbours):
    for r in range(G):
        colour = BAND(r / (G - 1))
        ax.bar3d(0, 0, r * 0.55, 4, 4, 0.12, color=colour, alpha=0.35, edgecolor=colour, linewidth=0.4, shade=False)
    mark(ax, on_cube, neighbours)


GEOMETRIES = (
    ("Torus", "both axes wrap: the highest band's row meets the lowest",
     _grid_neighbours(True, True), True, True, on_torus, "surface", (48, -35)),
    ("Cylinder", "columns wrap; the frequency axis is open, as in the cochlea",
     _grid_neighbours(False, True), False, True, on_cylinder, "surface", (18, -60)),
    ("Sheet", "nothing wraps; coupling stops at every edge",
     _grid_neighbours(False, False), False, False, on_sheet, "surface", (18, -70)),
    ("Sphere", "rows are latitudes, poles open; columns are longitudes that wrap",
     _grid_neighbours(False, True), False, True, on_sphere, "surface", (15, -60)),
    ("Helix", "all 256 sites in one closed ring, four rows (one octave) per turn",
     _helix_neighbours(), None, None, on_helix, "helix", (15, -60)),
    ("Cube", "each row's 16 columns fold into a 4 × 4 slab; all three axes wrap",
     _cube_neighbours(), True, None, on_cube, "cube", (20, -55)),
    ("Coil", "all 256 sites in one open line, apex (lowest band) to base (highest), one octave per turn",
     _coil_neighbours(), None, None, on_coil, "coil", (35, -60)),
    ("Cochlea", "the coil; influence runs base to apex three to one, and coupling grows with curvature toward the apex",
     _coil_neighbours(), None, None, on_coil, "cochlea", (35, -60)),
)

NOTES = {
    "Helix": "each row runs on into the next; one turn joins bands an octave apart (±64 sites)",
    "Cube": "columns 0–3, 4–7, 8–11 and 12–15 are the four lines of each row's slab",
    "Coil": "each row runs on into the next; one turn (±64 sites) joins adjacent turns; the ends never meet",
}


def main() -> None:
    fig = plt.figure(figsize=(12.5, 15.3))
    grid = fig.add_gridspec(4, 4, width_ratios=[1, 1.25, 1, 1.25], hspace=0.22, wspace=0.05)
    for i, (name, blurb, nb, glue_rows, glue_cols, place, kind, view) in enumerate(GEOMETRIES):
        row, col = divmod(i, 2)
        ax_flat = fig.add_subplot(grid[row, 2 * col])
        ax_3d = fig.add_subplot(grid[row, 2 * col + 1], projection="3d")
        note = "\n".join(textwrap.wrap(blurb + (f"; {NOTES[name]}" if name in NOTES else ""), 46))
        flat(ax_flat, nb, glue_rows, glue_cols, note, cube=(kind == "cube"))
        if kind == "surface":
            surface(ax_3d, place)
            mark(ax_3d, place, nb)
        elif kind == "helix":
            draw_helix(ax_3d, nb)
        elif kind in ("coil", "cochlea"):
            draw_coil(ax_3d, nb, cochlea=(kind == "cochlea"))
        else:
            draw_cube(ax_3d, nb)
        shape_axes(ax_3d, view)
        ax_flat.set_title(name, fontsize=11, fontweight="bold", loc="left", pad=6)
    fig.text(0.5, 0.02, "Left: a channel's 16 × 16 grid; glued edges share a colour and an arrow, dark edges are open, grey edges are explained below the grid. "
             "Right: the shape the gluing makes.\nColour: the mel band that drives each row (lowest dark, highest "
             "yellow). Star: one oscillator. Dots: its nearest neighbours under that geometry.",
             ha="center", fontsize=9, color=INK, linespacing=1.5)
    stem = FIGURES_DIR / "a2-lattice-geometries"
    for suffix, kwargs in ((".pdf", {"metadata": {"CreationDate": None, "Creator": None, "Producer": None}}),
                           (".png", {"dpi": 200, "metadata": {"Software": None}})):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    print(f"wrote {stem}.{{pdf,png}}")


if __name__ == "__main__":
    main()

"""Draw the appendix schematic of the six coupling functions.

    cd src && uv run python ../scripts/draw_coupling_figure.py      # from the paper's folder

A drawing, not a result: nothing here reads the record. Each coupling function
is drawn twice for one neighbour j with a positive kernel weight: on the left,
the push an oscillator would feel at each phase around the circle while j sits
at the top; on the right, that push against the phase difference. The forms and
constants follow harness/models/phase.py and stuart_landau.py (alpha = pi/4,
beta = 0.5). Writes resources/figures/a3-coupling-functions.{pdf,png}.
"""

from __future__ import annotations

import math
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harness.utils.paths import FIGURES_DIR

ALPHA, BETA = math.pi / 4, 0.5
THETA_J = math.pi / 2                      # the neighbour sits at the top of the circle
INK, MUTED, PUSH, REF = "#222222", "#8C8C8C", "#534AB7", "#D85A30"


# ---------------------------------------------------------------------------
# The push on an oscillator at phase theta from a neighbour at phase theta_j:
# (along the circle, toward the centre's opposite). Radial is zero for phase cores.
# ---------------------------------------------------------------------------

def kuramoto(theta, theta_j):
    return math.sin(theta_j - theta), 0.0


def sakaguchi(theta, theta_j):
    return math.sin(theta_j - theta - ALPHA), 0.0


def harmonic2(theta, theta_j):
    d = theta_j - theta
    return math.sin(d) + BETA * math.sin(2 * d), 0.0


def winfree(theta, theta_j):
    return -math.sin(theta) * (1 + math.cos(theta_j)), 0.0


def stuart_landau(theta, theta_j):
    """K (z_j - z_i) at unit amplitude: along the circle sin(d), outward cos(d) - 1 (never positive)."""
    d = theta_j - theta
    return math.sin(d), math.cos(d) - 1


def fixed_amplitude(theta, theta_j):
    return math.sin(theta_j - theta), 0.0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def circle(ax, push, marks=()):
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=1)
    for theta in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        along, out = push(theta, THETA_J)
        tangent, radial = np.array([-math.sin(theta), math.cos(theta)]), np.array([math.cos(theta), math.sin(theta)])
        vec = 0.28 * (along * tangent + out * radial)
        start = radial
        if np.hypot(*vec) > 0.02:
            ax.annotate("", xy=start + vec, xytext=start,
                        arrowprops={"arrowstyle": "-|>", "color": PUSH, "lw": 1.3, "mutation_scale": 8})
        else:
            ax.scatter(*start, s=6, color=PUSH, zorder=4)
    ax.scatter([math.cos(THETA_J)], [math.sin(THETA_J)], s=70, color=INK, zorder=6)
    ax.text(math.cos(THETA_J), math.sin(THETA_J) + 0.2, "neighbour j", ha="center", va="bottom", fontsize=7)
    for angle, label in marks:
        x, y = math.cos(angle), math.sin(angle)
        ax.scatter([x], [y], s=55, facecolor="white", edgecolor=REF, lw=1.5, zorder=6)
        ha = "left" if x > 0.3 else "right" if x < -0.3 else "center"
        ax.text(1.2 * x + (0.08 if ha == "left" else -0.08 if ha == "right" else 0), 1.25 * y, label,
                ha=ha, va="center", fontsize=7, color=REF)
    ax.scatter([0], [0], s=4, color=MUTED)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.6, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")


def curve(ax, series, xlabel, points=()):
    x = np.linspace(-np.pi, np.pi, 400)
    for fn, label, style in series:
        ax.plot(x, [fn(v) for v in x], label=label, **style)
    ax.axhline(0, color=MUTED, lw=0.6)
    for at, kind in points:
        face = INK if kind == "stable" else "white"
        ax.scatter([at], [0], s=30, facecolor=face, edgecolor=INK, zorder=5, lw=1)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0", "π/2", "π"], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel("push on i", fontsize=7)
    ax.set_ylim(-2.3, 2.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")


SOLID = {"color": PUSH, "lw": 1.6}
DASHED = {"color": MUTED, "lw": 1.0, "ls": "--"}
DIFF = "phase difference θⱼ − θᵢ"

FUNCTIONS = (
    ("Kuramoto", "Kᵢⱼ sin(θⱼ − θᵢ)",
     "each oscillator is pulled along the circle toward its neighbour's phase: at rest in phase, "
     "pushed away from opposite phase. The reference.",
     kuramoto, (),
     [(lambda d: math.sin(d), "sin(θⱼ − θᵢ)", SOLID)], DIFF,
     [(0, "stable"), (-math.pi, "unstable"), (math.pi, "unstable")]),
    ("Kuramoto–Sakaguchi", "Kᵢⱼ sin(θⱼ − θᵢ − α),  α = π/4",
     "the pull aims α behind the neighbour: an in-phase pair still locks, but slowed, "
     "and a population can settle into partial coherence rather than all-or-nothing locking.",
     sakaguchi, ((THETA_J - ALPHA, "aim"),),
     [(lambda d: math.sin(d - ALPHA), "sin(θⱼ − θᵢ − α)", SOLID), (lambda d: math.sin(d), "Kuramoto", DASHED)], DIFF,
     [(ALPHA, "stable")]),
    ("Second harmonic", "Kᵢⱼ [sin(θⱼ − θᵢ) + β sin 2(θⱼ − θᵢ)],  β = 0.5",
     "adds a second harmonic: the pull toward the neighbour grows, and the push away from opposite phase "
     "vanishes, so pairs half a cycle apart are no longer driven apart.",
     harmonic2, ((THETA_J + math.pi, "no push"),),
     [(lambda d: math.sin(d) + BETA * math.sin(2 * d), "with second harmonic", SOLID),
      (lambda d: math.sin(d), "Kuramoto", DASHED)], DIFF,
     [(0, "stable"), (-math.pi, "neutral"), (math.pi, "neutral")]),
    ("Winfree", "−Kᵢⱼ sin θᵢ (1 + cos θⱼ)",
     "every push points toward phase 0: the oscillator's own phase sets how much it responds, "
     "and the neighbour's phase only how strong the push is.",
     winfree, ((0.0, "phase 0"),),
     [(lambda p: -math.sin(p), "sensitivity −sin θᵢ (own phase)", SOLID),
      (lambda p: 1 + math.cos(p), "influence 1 + cos θⱼ (neighbour's)", {"color": REF, "lw": 1.4})],
     "phase", ()),
    ("Stuart–Landau", "Kᵢⱼ (zⱼ − zᵢ),  z = r e^(iθ), r the amplitude",
     "each oscillator is pulled straight toward its neighbour in the plane: along the circle, as Kuramoto, "
     "and inward, so neighbours out of phase shrink each other's amplitude; amplitude then relaxes back to 1.",
     stuart_landau, (),
     [(lambda d: math.sin(d), "along the circle: sin(θⱼ − θᵢ)", SOLID),
      (lambda d: math.cos(d) - 1, "amplitude: cos(θⱼ − θᵢ) − 1", {"color": REF, "lw": 1.4})], DIFF,
     [(0, "stable")]),
    ("Stuart–Landau, fixed amplitude", "the part of Kᵢⱼ (zⱼ − zᵢ) along the circle, |z| = 1",
     "the amplitude is held at 1, so only the pull along the circle is left: Kuramoto again. "
     "It isolates what amplitude adds.",
     fixed_amplitude, (),
     [(lambda d: math.sin(d), "along the circle: sin(θⱼ − θᵢ)", SOLID),
      (lambda d: math.cos(d) - 1, "amplitude part, removed", DASHED)], DIFF,
     [(0, "stable"), (-math.pi, "unstable"), (math.pi, "unstable")]),
)


def main() -> None:
    fig = plt.figure(figsize=(12.5, 13.5))
    outer = fig.add_gridspec(3, 2, hspace=0.12, wspace=0.12)
    for i, (name, formula, blurb, push, marks, series, xlabel, points) in enumerate(FUNCTIONS):
        cell = outer[divmod(i, 2)].subgridspec(3, 2, height_ratios=[0.2, 1, 0.36], width_ratios=[1, 1.2],
                                               hspace=0.05, wspace=0.3)
        head = fig.add_subplot(cell[0, :])
        head.axis("off")
        head.text(0, 0.95, name, fontsize=11, fontweight="bold", va="top")
        head.text(0, 0.2, formula, fontsize=8.5, color=INK, va="top")
        circle(fig.add_subplot(cell[1, 0]), push, marks)
        curve(fig.add_subplot(cell[1, 1]), series, xlabel, points)
        foot = fig.add_subplot(cell[2, :])
        foot.axis("off")
        foot.text(0, 0.42, "\n".join(textwrap.wrap(blurb, 84)), fontsize=7.5, color=MUTED, va="top")
    fig.text(0.5, 0.06, "Left: the push an oscillator would feel at each phase around the circle from one neighbour j "
             "(black, at the top) with a positive kernel weight; arrows point the way it is pushed.\n"
             "Right: that push against the phase difference; filled dots are pairs at rest and stable, open dots at "
             "rest but unstable or neutral. A negative weight reverses every arrow, and each oscillator\n"
             "feels the sum over every other oscillator in its channel.",
             ha="center", fontsize=8.5, color=INK, linespacing=1.5)
    stem = FIGURES_DIR / "a3-coupling-functions"
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 200})):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    print(f"wrote {stem}.{{pdf,png}}")


if __name__ == "__main__":
    main()

"""The figures, and the tables printed beside them, drawn from the record.

    uv run python -m harness.experiment.figures      # write the figures, print the paper's tables

Every number in a figure and in its table comes from the same accuracies that
`harness.experiment.summary` reports, so the paper cannot drift from the record.
"""

from __future__ import annotations

import argparse

import torch

from harness.experiment import record as rec
from harness.experiment import summary as sm
from harness.experiment import terms
from harness.utils.paths import FIGURES_DIR

#: how every figure is saved: vector and raster, with no build date, tool or version in either file,
#: so a figure carries nothing that dates or identifies the machine that drew it, and redrawing it
#: from the same record gives the same bytes
SAVE = ((".pdf", {"metadata": {"CreationDate": None, "Creator": None, "Producer": None}}),
        (".png", {"dpi": 200, "metadata": {"Software": None}}))

NOISES = tuple((n, sm.snr(n)) for n in (None, 0.0, 5.0))

#: the controls experiment's recognition arms: (arm, read, table label, legend label, colour); gain is set per figure
BASELINE = ("baseline", "windowed@wholeclip",
         "**Spectrogram-only baseline**: the readout reads the 16 mel band energies directly; no reservoir",
         "Spectrogram-only baseline: no reservoir", "#8C8C8C")
COUPLED = (sm.COUPLED, "windowed", "**Coupled oscillator network**{gain}: 1,024 Kuramoto oscillators, untrained",
         "Coupled oscillator network, Kuramoto{gain}", "#534AB7")
UNCOUPLED = (sm.UNCOUPLED, "windowed", "**Uncoupled oscillator network**{gain}: the same network with its coupling removed",
             "Uncoupled oscillator network, Kuramoto{gain}", "#A9A4DB")
BANK_STATE = ("bank-state", "windowed",
              "**Leaky-integrator bank, state-matched**{gain}: 1,024 leaky integrators with the network's states and "
              "parameters, untrained",
              "Leaky-integrator bank, state-matched{gain}", "#D85A30")
TRAINED_REFERENCE = ("trained-gru", "windowed", "**GRU**: a trained baseline, 1,944 parameters, trained end to end",
                     "GRU: trained baseline", "#0F6E56")
DYNAMICAL = (COUPLED, UNCOUPLED, BANK_STATE)


def _bars(gains: tuple[float, ...]) -> list[tuple]:
    """(arm, read, gain, table label, legend label, colour, hatched) per bar, in plotting order.

    Input gain applies only to the reservoirs; the spectrogram-only baseline
    and the GRU appear once, without it.
    """
    def one(spec, gain):
        arm, read, table, legend, colour = spec
        tag = "" if gain is None else f" (gain = {gain:g})"
        return (arm, read, gain, table.format(gain=tag), legend.format(gain=tag), colour, gain == 2.0)
    return ([one(BASELINE, None)] + [one(spec, g) for spec in DYNAMICAL for g in gains]
            + [one(TRAINED_REFERENCE, None)])


#: the figures: file stem, the gains shown
FIGURES = (("c1-recognition-gain1", (1.0,)), ("c2-recognition-gain2", (2.0,)),
           ("c3-recognition-both-gains", (1.0, 2.0)))


def recognition_cells() -> dict[tuple, dict]:
    """(arm, read, gain, noise) -> the accuracy record at the primary cell."""
    cells = rec.load(["controls-recognition"])
    return {(r["arm"], r["read"], r["gain"], r["noise"]): r for r in sm.accuracies(cells)
            if r["width"] == rec.PRIMARY_WIDTH and r["n_train"] == rec.PRIMARY_SIZE}


def recognition_table(acc: dict[tuple, dict], gains: tuple[float, ...]) -> str:
    lines = ["| arm | " + " | ".join(h for _, h in NOISES) + " |", "|---" * (len(NOISES) + 1) + "|"]
    for arm, read, gain, label, *_ in _bars(gains):
        cells = [acc[(arm, read, gain, noise)] for noise, _ in NOISES]
        lines.append(f"| {label} | " + " | ".join(f"{c['mean']:.1f} ± {c['sd']:.1f}" for c in cells) + " |")
    return "\n".join(lines)


def recognition_figure(acc: dict[tuple, dict], stem: str, gains: tuple[float, ...]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bars = _bars(gains)
    n = len(bars)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(7.5 if n <= 5 else 9.5, 4.6 if n <= 5 else 5.0))
    for i, (arm, read, gain, _, legend, colour, hatched) in enumerate(bars):
        cells = [acc[(arm, read, gain, noise)] for noise, _ in NOISES]
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(NOISES))]
        means, sds = [c["mean"] for c in cells], [c["sd"] for c in cells]
        ax.bar(xs, means, width, yerr=sds, capsize=2.5, color=colour, label=legend,
               hatch="///" if hatched else None, edgecolor="white" if hatched else None, linewidth=0,
               error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
        for x, m, s in zip(xs, means, sds, strict=True):
            ax.text(x, m + s + 1.2, f"{m:.1f}", ha="center", va="bottom", rotation=90,
                    fontsize=7 if n <= 5 else 6)
    ax.axhline(10, color="#333333", linewidth=0.8, linestyle=":")
    ax.text(len(NOISES) - 0.5, 11, "chance", ha="right", va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(range(len(NOISES)), [h for _, h in NOISES])
    ax.set_xlabel("noise level of the training and test audio")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2 if n <= 5 else 3)
    fig.tight_layout()
    # the LaTeX builds take the vector PDF; everything else takes the PNG
    path = FIGURES_DIR / stem
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in SAVE:
        fig.savefig(path.with_suffix(suffix), **kwargs)
    plt.close(fig)
    print(f"wrote {path}.{{pdf,png}}")


#: the design experiment's conditions, in plotting order: (noise, gain, legend, colour, marker)
CONDITIONS = tuple((n, g, f"{sm.snr(n)}, gain = {g:g}", colour, marker) for n, g, colour, marker in (
    (0.0, 1.0, "#534AB7", "o"), (0.0, 2.0, "#A9A4DB", "o"), (5.0, 1.0, "#D85A30", "s"), (5.0, 2.0, "#EFA98F", "s")))


def design_differences() -> list[dict]:
    """The design experiment's level-minus-reference differences at the primary cell, in the summary's order."""
    cells = rec.load(None)
    return [r for r in sm.design(cells) if r["width"] == rec.PRIMARY_WIDTH and r["n_train"] == rec.PRIMARY_SIZE
            and "ci95" in r and "baseline" not in r["comparison"]]


def design_table(rows: list[dict]) -> str:
    """Every design difference in every condition, mean ± standard deviation over seeds; the intervals are
    in the figure."""
    names = list(dict.fromkeys(r["comparison"] for r in rows))
    by = {(r["comparison"], r["noise"], r["gain"]): r for r in rows}
    lines = ["| level minus reference | " + " | ".join(c[2] for c in CONDITIONS) + " |",
             "|---" * (len(CONDITIONS) + 1) + "|"]
    for name in names:
        cells = [by[(name, noise, gain)] for noise, gain, *_ in CONDITIONS]
        lines.append(f"| {name.split(': ', 1)[1]} | " + " | ".join(
            f"{c['mean']:+.2f} ± {c['sd']:.2f}".replace("-", "−") for c in cells) + " |")
    return "\n".join(lines)


def design_figure(rows: list[dict], stem: str = "c4-design-differences") -> None:
    """Each design level minus its reference: the mean over seeds and the paired 95% interval, per condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(r["comparison"] for r in rows))
    by = {(r["comparison"], r["noise"], r["gain"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(8.0, 0.42 * len(names) + 1.4))
    for i, (noise, gain, legend, colour, marker) in enumerate(CONDITIONS):
        offset = (i - 1.5) * 0.17
        for j, name in enumerate(names):
            r = by[(name, noise, gain)]
            y = j + offset
            ax.plot(r["ci95"], [y, y], color=colour, linewidth=1.2)
            ax.plot(r["mean"], y, marker=marker, color=colour, markersize=4.5, linestyle="none",
                    label=legend if j == 0 else None)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle=":")
    ax.set_yticks(range(len(names)), [n.split(": ", 1)[1] for n in names], fontsize=8)
    for j, name in enumerate(names):
        if j == 0 or name.split(":")[0] != names[j - 1].split(":")[0]:
            ax.text(1.01, j, name.split(":")[0], transform=ax.get_yaxis_transform(), fontsize=7.5,
                    color="#555555", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("difference in test accuracy (points), level minus reference")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.9 / (0.42 * len(names) + 1.4)),
              ncol=4)
    fig.tight_layout()
    path = FIGURES_DIR / stem
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in SAVE:
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {path}.{{pdf,png}}")


#: the training-size figure's arms: (arm, read, gain, legend, colour); the reservoirs at gain 1
SIZE_ARMS = (("baseline", "windowed@wholeclip", None, "spectrogram-only baseline, whole clip", "#8C8C8C"),
             (sm.COUPLED, "windowed", 1.0, "coupled oscillator network, Kuramoto", "#534AB7"),
             (sm.UNCOUPLED, "windowed", 1.0, "uncoupled oscillator network, Kuramoto", "#A9A4DB"),
             ("bank-state", "windowed", 1.0, "leaky-integrator bank, state-matched", "#D85A30"),
             ("trained-transformer", "windowed", None, "transformer", "#0F6E56"),
             ("trained-s4d", "windowed", None, "S4D", "#5DB39A"),
             ("trained-gru", "windowed", None, "GRU", "#A7D8C8"))


def size_figure(stem: str = "c5-training-size") -> None:
    """Accuracy against training-set size at widths 192 (solid) and 4,096 (dashed), per noise level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = rec.load(["controls-recognition"])
    acc = {(r["arm"], r["read"], r["gain"], r["noise"], r["width"], r["n_train"]): r for r in sm.accuracies(cells)}
    sizes = sorted({k[5] for k in acc})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True)
    for ax, noise, title in ((axes[0], 0.0, sm.snr(0.0)), (axes[1], 5.0, sm.snr(5.0))):
        for arm, read, gain, legend, colour in SIZE_ARMS:
            for width, style in ((192, "-"), (4096, "--")):
                rows = [acc.get((arm, read, gain, noise, width, n)) for n in sizes]
                if any(r is None for r in rows) or (width == 4096 and arm in ("baseline",) + TRAINED):
                    continue
                ax.errorbar(sizes, [r["mean"] for r in rows], yerr=[r["sd"] for r in rows], color=colour,
                            linestyle=style, marker="o", markersize=3.5, capsize=2, linewidth=1.2,
                            label=legend if width == 192 else None)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{n:,}" for n in sizes])
        ax.minorticks_off()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training clips")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("task accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    path = FIGURES_DIR / stem
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in SAVE:
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {path}.{{pdf,png}}")


#: the gain figure's coupling functions and colours
GAIN_COUPLINGS = (("kuramoto", "#534AB7"), ("kuramoto-sakaguchi", "#8C7FD6"), ("second-harmonic", "#2F6DB5"),
                  ("winfree", "#5DB39A"), ("stuart-landau", "#D85A30"), ("stuart-landau-fixed", "#EFA98F"))


def gain_figure(stem: str = "c6-gain-sweep") -> None:
    """Accuracy against input gain for every coupling function at the reference configuration: the
    design experiment's gains 1 and 2 with the sweep's 3 to 12, per noise level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from harness.experiment import plan
    from harness.experiment.arms import Arm
    groups = [f"design-recognition-{c}" for c, _ in GAIN_COUPLINGS]
    groups += [f"sweep-recognition-{c}" for c, _ in GAIN_COUPLINGS]
    acc = {(r["arm"], r["noise"], r["gain"]): r for r in sm.accuracies(rec.load(groups))
           if r["width"] == rec.PRIMARY_WIDTH and r["n_train"] == rec.PRIMARY_SIZE and r["read"] == "windowed"}
    gains = tuple(sorted((1.0, 2.0) + plan.SWEEP_GAINS))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for ax, noise, title in ((axes[0], 0.0, sm.snr(0.0)), (axes[1], 5.0, sm.snr(5.0))):
        for coupling, colour in GAIN_COUPLINGS:
            label = Arm("network", coupling=coupling).label()
            pts = [(g, acc[(label, noise, g)]) for g in gains if (label, noise, g) in acc]
            if pts:
                ax.errorbar([g for g, _ in pts], [r["mean"] for _, r in pts], yerr=[r["sd"] for _, r in pts],
                            color=colour, marker="o", markersize=3.5, capsize=2, linewidth=1.2,
                            label=terms.level(coupling))
        ax.set_xscale("log")
        ax.set_xticks(gains, [f"{g:g}" for g in gains])
        ax.minorticks_off()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("input gain")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    path = FIGURES_DIR / stem
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in SAVE:
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {path}.{{pdf,png}}")


#: the arms figures' rows, top to bottom: (arm label, read suffix, row label, colour, group); the read is the
#: task's primary one, plus the suffix. Gain applies to the reservoirs only: gain 1 filled, gain 2 open.
SL_REFERENCE = "coupled-stuart-landau-torus-random-restoring0.3-ceiling1"
ARM_ROWS = (
    ("baseline", "@wholeclip", "Spectrogram-only baseline, whole clip", "#6E6E6E", "input"),
    ("baseline", "", "Spectrogram-only baseline, from frame 16", "#ABABAB", "input"),
    (sm.COUPLED, "", "Coupled oscillator network, Kuramoto", "#534AB7", "network"),
    (SL_REFERENCE, "", "Coupled oscillator network, Stuart–Landau", "#8E2C8A", "network"),
    (sm.UNCOUPLED, "", "Uncoupled oscillator network, Kuramoto", "#A9A4DB", "network"),
    ("bank-state", "", "Leaky-integrator bank, state-matched", "#D85A30", "bank"),
    ("bank-width", "", "Leaky-integrator bank, width-matched", "#EFA98F", "bank"),
    *((f"trained-{a}", "", name, "#0F6E56", "trained") for a, name in
      (("gru", "GRU"), ("tcn", "TCN"), ("cnn", "CNN"), ("transformer", "Transformer"), ("s4d", "S4D"))),
)


def arms_accuracy(task: str) -> dict[tuple, dict]:
    """(arm, read, gain, noise) -> accuracy at the primary width and training size; the order task's averaged
    over its five digit pairs within each seed. Recognition adds the Stuart–Landau network's reference cells."""
    if task == "recognition":
        rows = sm.accuracies(rec.load(["controls-recognition", "design-recognition-stuart-landau"]))
    else:
        rows = sm.accuracies(rec.load(["controls-order"]))
    rows = [r for r in rows if r["width"] == rec.PRIMARY_WIDTH and r["n_train"] == rec.PRIMARY_SIZE
            and r.get("projection", "fixed") == "fixed"]
    if task == "order":
        rows = sm._pooled(rows)
    return {(r["arm"], r["read"], r["gain"], r["noise"]): r for r in rows}


def arms_figure(task: str, stem: str) -> None:
    """Every arm's accuracy per noise level: the mean over three seeds and one standard deviation, reservoirs
    at gain 1 (filled) and 2 (open)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    acc = arms_accuracy(task)
    read = sm.READ[task]
    rows = [r for r in ARM_ROWS if any(k[0] == r[0] and k[1] == read + r[1] for k in acc)]
    fig, axes = plt.subplots(1, len(NOISES), figsize=(10.0, 0.3 * len(rows) + 1.3), sharey=True)
    for ax, (noise, title) in zip(axes, NOISES, strict=True):
        lo, hi = 100.0, 0.0
        for y, (arm, suffix, _, colour, _) in enumerate(rows):
            for gain, dy, face in ((None, 0.0, colour), (1.0, -0.14, colour), (2.0, 0.14, "white")):
                r = acc.get((arm, read + suffix, gain, noise))
                if r is None:
                    continue
                ax.errorbar(r["mean"], y + dy, xerr=r["sd"] or 0, fmt="o", color=colour, markerfacecolor=face,
                            markersize=5, capsize=2, elinewidth=1, markeredgewidth=1.1)
                lo, hi = min(lo, r["mean"] - (r["sd"] or 0)), max(hi, r["mean"] + (r["sd"] or 0))
        for y in range(1, len(rows)):
            if rows[y][4] != rows[y - 1][4]:
                ax.axhline(y - 0.5, color="#DDDDDD", linewidth=0.8)
        base = acc.get(("baseline", read + "@wholeclip", None, noise))
        if base is not None and task == "recognition":
            ax.axvline(base["mean"], color="#6E6E6E", linewidth=0.8, linestyle="--", alpha=0.6)
        if task == "order":
            ax.axvline(50, color="#333333", linewidth=0.8, linestyle=":")
            ax.set_xlim(45, 101)
        else:
            pad = 0.06 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("task accuracy (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(rows)), [r[2] for r in rows], fontsize=8.5)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], marker="o", linestyle="none", color="#555555", label="reservoir at gain 1, or an arm without gain"),
               Line2D([], [], marker="o", linestyle="none", color="#555555", markerfacecolor="white", label="reservoir at gain 2")]
    if task == "order":
        handles.append(Line2D([], [], color="#333333", linewidth=0.8, linestyle=":", label="chance"))
    else:
        handles.append(Line2D([], [], color="#6E6E6E", linewidth=0.8, linestyle="--", alpha=0.6,
                              label="spectrogram-only baseline, whole clip"))
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, stem)


def _save(fig, stem: str) -> None:
    import matplotlib.pyplot as plt
    path = FIGURES_DIR / stem
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in SAVE:
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {path}.{{pdf,png}}")


#: the design figure's panels: (title, factor prefix, reference, [(level as the summary names it, row label)])
DESIGN_PANELS = (
    ("coupling function, minus Kuramoto", "coupling function", "Kuramoto",
     (("Kuramoto–Sakaguchi", "Kuramoto–Sakaguchi"), ("second harmonic", "second harmonic"),
      ("Stuart–Landau", "Stuart–Landau"), ("Stuart–Landau, fixed amplitude", "Stuart–Landau, fixed amplitude"),
      ("Winfree", "Winfree"))),
    ("natural frequencies, minus random", "natural frequencies", "random",
     (("identical", "identical"), ("tonotopic", "tonotopic"))),
    ("lattice geometry, minus torus", "lattice geometry", "torus",
     (("cube", "cube"), ("cylinder", "cylinder"), ("helix", "helix"), ("sheet", "sheet"), ("sphere", "sphere"),
      ("coil", "coil"), ("cochlea", "cochlea"), ("cochlea-matched", "cochlea, matched coupling"))),
)


def design_panels_figure(stem: str = "c4-design-differences") -> None:
    """Each coupling function, natural frequencies and lattice geometry minus its reference, per condition: the
    mean over seeds and the paired 95% interval. The coil and cochlea come from the cochlea experiment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = rec.load(None)
    rows = [r for r in sm.design(cells) + sm.cochlea(cells)
            if r["width"] == rec.PRIMARY_WIDTH and r["n_train"] == rec.PRIMARY_SIZE and "ci95" in r]
    by = {(r["comparison"], r["noise"], r["gain"]): r for r in rows}
    fig = plt.figure(figsize=(10.0, 4.3))
    grid = fig.add_gridspec(2, 2, height_ratios=(5, 2.6), hspace=0.55, wspace=0.55)
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1]))
    for ax, (title, factor, ref, levels) in zip(axes, DESIGN_PANELS, strict=True):
        for i, (noise, gain, legend, colour, marker) in enumerate(CONDITIONS):
            offset = (i - 1.5) * 0.17
            for j, (level, _) in enumerate(levels):
                r = by.get((f"{factor}: {level} minus {ref}", noise, gain))
                if r is None:
                    continue
                ax.plot(r["ci95"], [j + offset] * 2, color=colour, linewidth=1.2)
                ax.plot(r["mean"], j + offset, marker=marker, color=colour, markersize=4, linestyle="none",
                        label=legend if (j == 0 and ax is axes[0]) else None)
        ax.axvline(0, color="#333333", linewidth=0.8, linestyle=":")
        if factor == "lattice geometry":
            ax.axhline(4.5, color="#DDDDDD", linewidth=0.8)
        ax.set_yticks(range(len(levels)), [label for _, label in levels], fontsize=8)
        ax.set_ylim(len(levels) - 0.5, -0.5)
        ax.set_title(title, fontsize=9.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in (axes[1], axes[2]):
        ax.set_xlabel("difference in test accuracy (points)", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    _save(fig, stem)


def _design_rows() -> dict[tuple, dict]:
    cells = rec.load(None)
    rows = [r for r in sm.design(cells) + sm.cochlea(cells)
            if r["width"] == rec.PRIMARY_WIDTH and r["n_train"] == rec.PRIMARY_SIZE and "ci95" in r]
    return {(r["comparison"], r["noise"], r["gain"]): r for r in rows}


def design_factor_figure(panel: int, stem: str, across: bool = False, by: dict | None = None) -> None:
    """One design factor's levels minus its reference, per condition: the mean over seeds and the paired 95%
    interval. Rows of levels by default; `across` lays the levels along the x-axis, for a short, wide figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = by if by is not None else _design_rows()
    _, factor, ref, levels = DESIGN_PANELS[panel]
    n = len(levels)
    if across:
        fig, ax = plt.subplots(figsize=(9.0, 2.9))
    else:
        height = 0.3 * n + 1.2
        fig, ax = plt.subplots(figsize=(7.0, height))
    for i, (noise, gain, legend, colour, marker) in enumerate(CONDITIONS):
        offset = (i - 1.5) * (0.17 if across else 0.18)
        for j, (level, _) in enumerate(levels):
            r = by.get((f"{factor}: {level} minus {ref}", noise, gain))
            if r is None:
                continue
            if across:
                ax.plot([j + offset] * 2, r["ci95"], color=colour, linewidth=1.2)
                ax.plot(j + offset, r["mean"], marker=marker, color=colour, markersize=4, linestyle="none",
                        label=legend if j == 0 else None)
            else:
                ax.plot(r["ci95"], [j + offset] * 2, color=colour, linewidth=1.2)
                ax.plot(r["mean"], j + offset, marker=marker, color=colour, markersize=3.5, linestyle="none",
                        label=legend if j == 0 else None)
    labels = [label.replace(", ", ",\n") if across else label for _, label in levels]
    if across:
        ax.axhline(0, color="#333333", linewidth=0.8, linestyle=":")
        if factor == "lattice geometry":
            ax.axvline(4.5, color="#DDDDDD", linewidth=0.8)
        ax.set_xticks(range(n), labels, fontsize=8.5)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylabel(f"minus {ref} (% points)", fontsize=8.5)
    else:
        ax.axvline(0, color="#333333", linewidth=0.8, linestyle=":")
        ax.set_yticks(range(n), labels, fontsize=8.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xlabel(f"difference in task accuracy, minus {ref} (% points)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    handles, legends = ax.get_legend_handles_labels()
    fig.legend(handles, legends, frameon=False, fontsize=7.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.4 / (2.9 if across else height), 1, 1))
    _save(fig, stem)


#: the design experiment's factors as figures of their own: (panel in DESIGN_PANELS, file stem, levels across)
DESIGN_FIGURES = ((0, "c4-coupling-functions", False), (1, "c4-natural-frequencies", False),
                  (2, "c4-lattice-geometries", True))


def sweep_figure(stem: str = "c6-restoring-ceiling-gain") -> None:
    """Accuracy against restoring strength, coupling ceiling and input gain for every coupling function at the
    reference configuration, at 0 and -5 dB: the design experiment's levels with the sweep's, gain 1 in the
    first two columns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from harness.experiment import plan
    from harness.experiment.arms import Arm
    groups = [f"{e}-recognition-{c}" for e in ("design", "sweep") for c, _ in GAIN_COUPLINGS]
    acc = {(r["arm"], r["noise"], r["gain"]): r for r in sm.accuracies(rec.load(groups))
           if r["width"] == rec.PRIMARY_WIDTH and r["n_train"] == rec.PRIMARY_SIZE and r["read"] == "windowed"}
    columns = (
        ("restoring strength", (0.1, 0.3) + plan.SWEEP_RESTORINGS, lambda c, v: (Arm("network", coupling=c, restoring=v), 1.0), False),
        ("coupling ceiling", (0.5, 1.0) + plan.SWEEP_CEILINGS, lambda c, v: (Arm("network", coupling=c, ceiling=v), 1.0), False),
        ("input gain", tuple(sorted((1.0, 2.0) + plan.SWEEP_GAINS)), lambda c, v: (Arm("network", coupling=c), v), True),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.0), sharey="row", gridspec_kw={"width_ratios": (1, 1, 1.6)})
    for row, noise in enumerate((0.0, 5.0)):
        for col, (name, values, arm_of, log) in enumerate(columns):
            ax = axes[row, col]
            ax.axvline((0.3, 1.0, 1.0)[col], color="#BBBBBB", linewidth=0.8, linestyle=":")     # the reference setting
            for coupling, colour in GAIN_COUPLINGS:
                pts = []
                for v in values:
                    arm, gain = arm_of(coupling, v)
                    r = acc.get((arm.label(), noise, gain))
                    if r is not None:
                        pts.append((v, r))
                if pts:
                    ax.errorbar([v for v, _ in pts], [r["mean"] for _, r in pts], yerr=[r["sd"] for _, r in pts],
                                color=colour, marker="o", markersize=3, capsize=1.5, linewidth=1.1,
                                label=terms.level(coupling))
            if log:
                ax.set_xscale("log")
                ax.minorticks_off()
            ax.set_xticks(values, [f"{v:g}" for v in values], fontsize=7.5)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_title(f"{name}, {sm.snr(noise)}", fontsize=9.5)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 1:
                ax.set_xlabel(name, fontsize=8.5)
        axes[row, 0].set_ylabel("task accuracy (%)", fontsize=8.5)
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, stem)


def one_way_f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """One-way ANOVA F per feature: [n, ...] features, [n] labels -> [...] between-class over within-class
    variance, each over its degrees of freedom. A feature constant within every class reads 0."""
    classes = y.unique()
    n, k = len(y), len(classes)
    grand = x.mean(0)
    between = sum((y == c).sum() * (x[y == c].mean(0) - grand) ** 2 for c in classes) / (k - 1)
    within = sum(((x[y == c] - x[y == c].mean(0)) ** 2).sum(0) for c in classes) / (n - k)
    return torch.nan_to_num(between / within, nan=0.0, posinf=0.0)


def anova_maps(windows: int = 16, noise: float = 0.0, gain: float = 1.0, seed: int = 0,
               n: int = 2048) -> dict[str, torch.Tensor]:
    """[windows, bands] maps of the mean F over each band's signals, per arm, on Protocol A training clips.

    Each signal is averaged over each of `windows` equal spans of the whole padded clip (frames 0 to 61,
    warm-up included, so the input's onset shows), and each window mean is tested for a difference between
    the ten digits. Descriptive only: where an arm carries class information, not whether it reads better.
    """
    from harness.experiment import arms as am
    from harness.experiment import protocol as pr
    from harness.experiment import run as rn
    from harness.experiment.arms import Arm
    bank = pr.load_bank()
    out = {}
    for key, arm in (("baseline", Arm("baseline")), ("kuramoto", Arm("network")),
                     ("stuart-landau", Arm("network", coupling="stuart-landau")), ("bank", Arm("bank"))):
        spec = rn.Spec("controls", "recognition", "spectrogram", noise, gain if arm.uses_gain else None, seed, arm,
                       sizes=(n,), native_sizes=())
        clips = rn.assemble(spec, bank)
        model = am.build_untrained(arm, gain, seed)
        feats = []
        with torch.no_grad():
            for rows, _, where in rn.batches(spec, clips):
                if where.start >= n:
                    break
                sig = am.untrained_signals(arm, model, rows)                      # [B, T, D]
                spans = torch.tensor_split(torch.arange(sig.shape[1]), windows)
                feats.append(torch.stack([sig[:, s].mean(1) for s in spans], 1))  # [B, W, D]
        f = one_way_f(torch.cat(feats)[:n].double(), clips.labels[:n])            # [W, D]
        g = am.GRID
        if arm.kind == "baseline":
            out[key] = f
        elif arm.kind == "network":                                                # sin then cos, [C, G, G] each
            out[key] = f.view(windows, 2, -1, g, g).mean((1, 2, 4))
        else:                                                                      # [C, G, G]
            out[key] = f.view(windows, -1, g, g).mean((1, 3))
    return out


def anova_figure(maps: dict[str, torch.Tensor] | None = None, stem: str = "c7-anova-f") -> None:
    """Heat maps of the mean F per mel band and time window, one per arm, and the mean F per band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = anova_maps() if maps is None else maps
    names = {"baseline": "spectrogram (the input)", "kuramoto": "coupled network, Kuramoto",
             "stuart-landau": "coupled network, Stuart–Landau", "bank": "leaky-integrator bank, state-matched"}
    colours = {"baseline": "#8C8C8C", "kuramoto": "#534AB7", "stuart-landau": "#D85A30", "bank": "#0F6E56"}
    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(3.0 * (len(maps) + 1), 3.2))
    for ax, (key, m) in zip(axes, maps.items(), strict=False):
        im = ax.imshow(m.T.numpy(), origin="lower", aspect="auto", cmap="magma")
        ax.axvline(16 / 62 * m.shape[0] - 0.5, color="white", linewidth=0.8, linestyle=":")
        ax.set_title(names[key], fontsize=8)
        ax.set_xlabel("time window (1/16 of the clip)", fontsize=7)
        ax.set_ylabel("mel band", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    ax = axes[-1]
    for key, m in maps.items():
        ax.plot(range(m.shape[1]), m.mean(0).numpy(), marker="o", markersize=2.5, color=colours[key],
                label=names[key])
    ax.set_xlabel("mel band", fontsize=7)
    ax.set_ylabel("mean F over time windows", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(frameon=False, fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIGURES_DIR / stem
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in SAVE:
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {path}.{{pdf,png}}")


TRAINED = ("trained-gru", "trained-tcn", "trained-cnn", "trained-transformer", "trained-s4d")


def trained_table(acc: dict[tuple, dict]) -> str:
    """The trained baselines at the primary cell, with their parameter counts and training failures."""
    import json
    from collections import Counter

    from harness.experiment import run as rn
    record = json.loads((rn.record_root() / "controls-recognition.json").read_text())["runs"]
    params, failed, total = {}, Counter(), Counter()
    for run_id, entry in record.items():
        label = run_id.split("/")[-2]
        if label in TRAINED:
            params[label] = entry["arm_meta"]["trained_params"]
            if entry["spec"]["noise_db"] is not None:
                total[label] += 1
                failed[label] += not entry["health"]["healthy"]
    lines = ["| trained baseline | parameters | " + " | ".join(h for _, h in NOISES)
             + " | failed runs with noise |", "|---" * (len(NOISES) + 3) + "|"]
    for label in TRAINED:
        cells = [acc[(label, "windowed", None, noise)] for noise, _ in NOISES]
        lines.append(f"| {terms.arm(label)} | {params[label]:,} | "
                     + " | ".join(f"{c['mean']:.1f} ± {c['sd']:.1f}" for c in cells)
                     + f" | {failed[label]} of {total[label]} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    acc = recognition_cells()
    for stem, gains in FIGURES:
        recognition_figure(acc, stem, gains)
    for gains in ((1.0,), (2.0,)):
        print(f"\ngain {gains[0]:g}\n" + recognition_table(acc, gains))
    print("\ntrained baselines\n" + trained_table(acc))
    size_figure()
    arms_figure("recognition", "c3-recognition-arms")
    arms_figure("order", "c8-order-arms")
    design_panels_figure()
    by = _design_rows()
    for panel, stem, across in DESIGN_FIGURES:
        design_factor_figure(panel, stem, across, by)
    sweep_figure()
    rows = design_differences()
    if rows:
        print("\ndesign\n" + design_table(rows))


if __name__ == "__main__":
    main()

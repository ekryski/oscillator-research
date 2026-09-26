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


#: the design experiment's conditions, in plotting order: (noise, gain, legend, colour, marker)
CONDITIONS = tuple((n, g, f"{sm.snr(n)}, gain = {g:g}", colour, marker) for n, g, colour, marker in (
    (0.0, 1.0, "#534AB7", "o"), (0.0, 2.0, "#A9A4DB", "o"), (5.0, 1.0, "#D85A30", "s"), (5.0, 2.0, "#EFA98F", "s")))


#: the gain figure's coupling functions and colours
GAIN_COUPLINGS = (("kuramoto", "#534AB7"), ("kuramoto-sakaguchi", "#8C7FD6"), ("second-harmonic", "#2F6DB5"),
                  ("winfree", "#5DB39A"), ("stuart-landau", "#D85A30"), ("stuart-landau-fixed", "#EFA98F"))


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


#: the readout widths the width figure draws, each with its marker: (width, shape, face, size, offset within the row)
WIDTHS = ((192, "o", "white", 5, -0.24), (1024, "o", None, 4, 0.0), (4096, "s", None, 4.5, 0.24))


def width_figure(stem: str = "fig07-sec4-7-readout-width") -> None:
    """Every recognition arm's accuracy at readout widths 192, 1,024 and 4,096, per noise level: the mean over three
    seeds and one standard deviation, reservoirs at gain 1, 2,048 training clips. An arm no wider than 192 is read
    as it is at every width, so it gets the one marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cells = sm.accuracies(rec.load(["controls-recognition", "design-recognition-stuart-landau"]))
    acc = {(r["arm"], r["read"], r["gain"], r["noise"], r["width"]): r for r in cells
           if r["n_train"] == rec.PRIMARY_SIZE and r.get("projection", "fixed") == "fixed"}
    read = sm.READ["recognition"]
    rows = [r for r in ARM_ROWS if (r[0], r[1]) != ("baseline", "")]
    fig, axes = plt.subplots(1, len(NOISES), figsize=(10.0, 0.3 * len(rows) + 1.3), sharey=True)
    for ax, (noise, title) in zip(axes, NOISES, strict=True):
        lo, hi = 100.0, 0.0
        for y, (arm, suffix, _, colour, _) in enumerate(rows):
            gain = None if arm == "baseline" or arm.startswith("trained-") else 1.0
            narrow = acc.get((arm, read + suffix, gain, noise, rec.PRIMARY_WIDTH))
            for w, shape, face, size, dy in WIDTHS:
                r = acc.get((arm, read + suffix, gain, noise, w))
                if r is None:
                    continue
                if w != rec.PRIMARY_WIDTH and narrow is not None and r["mean"] == narrow["mean"]:
                    continue              # no wider than 192, so read as it is: one marker, drawn at the row's centre
                dy = dy if narrow is None or w != rec.PRIMARY_WIDTH or any(
                    acc.get((arm, read + suffix, gain, noise, v), narrow)["mean"] != narrow["mean"]
                    for v, *_ in WIDTHS) else 0.0
                ax.errorbar(r["mean"], y + dy, xerr=r["sd"] or 0, fmt=shape, color=colour, markerfacecolor=face or colour,
                            markersize=size, capsize=2, elinewidth=1, markeredgewidth=1.1)
                lo, hi = min(lo, r["mean"] - (r["sd"] or 0)), max(hi, r["mean"] + (r["sd"] or 0))
        for y in range(1, len(rows)):
            if rows[y][4] != rows[y - 1][4]:
                ax.axhline(y - 0.5, color="#DDDDDD", linewidth=0.8)
        base = acc.get(("baseline", read + "@wholeclip", None, noise, rec.PRIMARY_WIDTH))
        if base is not None:
            ax.axvline(base["mean"], color="#6E6E6E", linewidth=0.8, linestyle="--", alpha=0.6)
        pad = 0.06 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("task accuracy (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(rows)), [r[2] for r in rows], fontsize=8.5)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], marker=shape, linestyle="none", color="#555555", markerfacecolor=face or "#555555",
                      markersize=size, label=f"width {w:,}") for w, shape, face, size, _ in WIDTHS]
    handles.append(Line2D([], [], color="#6E6E6E", linewidth=0.8, linestyle="--", alpha=0.6,
                          label="spectrogram-only baseline, whole clip, width 192"))
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.55, -0.02))
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
    fig.tight_layout(rect=(0, 0.2 / (2.9 if across else height), 1, 1))
    _save(fig, stem)


#: the design experiment's factors as figures of their own: (panel in DESIGN_PANELS, file stem, levels across)
DESIGN_FIGURES = ((0, "fig03-sec4-3-coupling-functions", False), (1, "fig05-sec4-4-natural-frequencies", False),
                  (2, "fig04-sec4-4-lattice-geometries", True))


def sweep_figure(stem: str = "fig06-sec4-5-restoring-ceiling-gain") -> None:
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


def anova_figure(maps: dict[str, torch.Tensor] | None = None, stem: str = "fig11-appD2-class-information") -> None:
    """The mean F per band across the top, and below it a heat map per arm of the mean F per mel band and
    time window, two by two."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = anova_maps() if maps is None else maps
    names = {"baseline": "spectrogram (the input)", "kuramoto": "coupled network, Kuramoto",
             "stuart-landau": "coupled network, Stuart–Landau", "bank": "leaky-integrator bank, state-matched"}
    colours = {"baseline": "#8C8C8C", "kuramoto": "#534AB7", "stuart-landau": "#D85A30", "bank": "#0F6E56"}
    fig = plt.figure(figsize=(7.0, 7.6))
    grid = fig.add_gridspec(3, 2, height_ratios=(0.75, 1, 1))
    ax = fig.add_subplot(grid[0, :])
    for key, m in maps.items():
        ax.plot(range(m.shape[1]), m.mean(0).numpy(), marker="o", markersize=3, color=colours[key],
                label=names[key])
    ax.set_xlabel("mel band", fontsize=8)
    ax.set_ylabel("mean F over time windows", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    for i, (key, m) in enumerate(maps.items()):
        ax = fig.add_subplot(grid[1 + i // 2, i % 2])
        im = ax.imshow(m.T.numpy(), origin="lower", aspect="auto", cmap="magma")
        ax.axvline(16 / 62 * m.shape[0] - 0.5, color="white", linewidth=0.8, linestyle=":")
        ax.set_title(names[key], fontsize=8.5)
        ax.set_xlabel("time window (1/16 of the clip)", fontsize=8)
        ax.set_ylabel("mel band", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
    fig.tight_layout()
    _save(fig, stem)


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    arms_figure("recognition", "fig01-sec4-1-recognition-arms")
    arms_figure("order", "fig02-sec4-2-order-arms")
    width_figure()
    by = _design_rows()
    for panel, stem, across in DESIGN_FIGURES:
        design_factor_figure(panel, stem, across, by)
    sweep_figure()


if __name__ == "__main__":
    main()

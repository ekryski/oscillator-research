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

NOISES = tuple((n, sm.snr(n)) for n in (None, 0.0, 5.0))

#: Tier 1 recognition arms: (arm, read, table label, legend label, colour); gain is set per figure
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
    cells = rec.load(["tier1-recognition-spectrogram"])
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
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 200})):
        fig.savefig(path.with_suffix(suffix), **kwargs)
    plt.close(fig)
    print(f"wrote {path}.{{pdf,png}}")


#: Tier 2's conditions, in plotting order: (noise, gain, legend, colour, marker)
CONDITIONS = tuple((n, g, f"{sm.snr(n)}, gain = {g:g}", colour, marker) for n, g, colour, marker in (
    (0.0, 1.0, "#534AB7", "o"), (0.0, 2.0, "#A9A4DB", "o"), (5.0, 1.0, "#D85A30", "s"), (5.0, 2.0, "#EFA98F", "s")))


def design_differences() -> list[dict]:
    """Tier 2's level-minus-reference differences at the primary cell, in the summary's order."""
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
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 200})):
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

    cells = rec.load(["tier1-recognition-spectrogram"])
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
    axes[0].set_ylabel("test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    path = FIGURES_DIR / stem
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 200})):
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {path}.{{pdf,png}}")


#: the gain figure's coupling functions and colours
GAIN_COUPLINGS = (("kuramoto", "#534AB7"), ("kuramoto-sakaguchi", "#8C7FD6"), ("second-harmonic", "#2F6DB5"),
                  ("winfree", "#5DB39A"), ("stuart-landau", "#D85A30"), ("stuart-landau-fixed", "#EFA98F"))


def gain_figure(stem: str = "c6-gain-sweep") -> None:
    """Accuracy against input gain for every coupling function at the reference configuration: Tier 2's
    gains 1 and 2 with the sweep's 3 to 12, per noise level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from harness.experiment import plan
    from harness.experiment.arms import Arm
    groups = [f"tier2-recognition-spectrogram-{c}" for c, _ in GAIN_COUPLINGS]
    groups += [f"sweep-recognition-spectrogram-{c}" for c, _ in GAIN_COUPLINGS]
    acc = {(r["arm"], r["noise"], r["gain"]): r for r in sm.accuracies(rec.load(groups))
           if r["width"] == rec.PRIMARY_WIDTH and r["n_train"] == rec.PRIMARY_SIZE and r["read"] == "windowed"}
    gains = (1.0, 2.0) + plan.SWEEP_GAINS
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
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 200})):
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {path}.{{pdf,png}}")


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
        spec = rn.Spec("tier1", "recognition", "spectrogram", noise, gain if arm.uses_gain else None, seed, arm,
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
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 200})):
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {path}.{{pdf,png}}")


TRAINED = ("trained-gru", "trained-tcn", "trained-cnn", "trained-transformer", "trained-s4d")


def trained_table(acc: dict[tuple, dict]) -> str:
    """The trained baselines at the primary cell, with their parameter counts and training failures."""
    import json
    from collections import Counter

    from harness.experiment import run as rn
    record = json.loads((rn.record_root() / "tier1-recognition-spectrogram.json").read_text())["runs"]
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
    gain_figure()
    rows = design_differences()
    if rows:
        design_figure(rows)
        print("\ndesign\n" + design_table(rows))


if __name__ == "__main__":
    main()

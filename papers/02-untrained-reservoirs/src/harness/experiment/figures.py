"""The figures, and the tables printed beside them, drawn from the record.

    uv run python -m harness.experiment.figures      # write the figures, print the paper's tables

Every number in a figure and in its table comes from the same accuracies that
`harness.experiment.summary` reports, so the paper cannot drift from the record.
"""

from __future__ import annotations

import argparse

from harness.experiment import record as rec
from harness.experiment import summary as sm
from harness.experiment import terms
from harness.utils.paths import FIGURES_DIR

NOISES = ((None, "clean"), (0.0, "0 dB"), (5.0, "+5 dB"))

#: Tier 1 recognition arms: (arm, read, table label, legend label, colour); gain is set per figure
BASELINE = ("baseline", "windowed@wholeclip",
         "**Spectrogram-only baseline**: the readout reads the 16 mel band energies directly; no reservoir",
         "Spectrogram-only baseline: no reservoir", "#8C8C8C")
COUPLED = (sm.COUPLED, "windowed", "**Coupled oscillator network**{gain}: 1,024 oscillators, untrained",
         "Coupled oscillator network{gain}", "#534AB7")
UNCOUPLED = (sm.UNCOUPLED, "windowed", "**Uncoupled oscillator network**{gain}: the same network with its coupling removed",
             "Uncoupled oscillator network{gain}", "#A9A4DB")
BANK_STATE = ("bank-state", "windowed",
              "**Leaky-integrator bank, state-matched**{gain}: 1,024 leaky integrators with the network's states and "
              "parameters, untrained",
              "Leaky-integrator bank, state-matched{gain}", "#D85A30")
TRANSFORMER = ("trained-transformer", "windowed", "**Transformer**: a trained baseline, 1,968 parameters, trained end to end",
               "Transformer: trained baseline", "#0F6E56")
DYNAMICAL = (COUPLED, UNCOUPLED, BANK_STATE)


def _bars(gains: tuple[float, ...]) -> list[tuple]:
    """(arm, read, gain, table label, legend label, colour, hatched) per bar, in plotting order.

    Input gain applies only to the reservoirs; the spectrogram-only baseline
    and the transformer appear once, without it.
    """
    def one(spec, gain):
        arm, read, table, legend, colour = spec
        tag = "" if gain is None else f" (gain = {gain:g})"
        return (arm, read, gain, table.format(gain=tag), legend.format(gain=tag), colour, gain == 2.0)
    return ([one(BASELINE, None)] + [one(spec, g) for spec in DYNAMICAL for g in gains]
            + [one(TRANSFORMER, None)])


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
CONDITIONS = ((0.0, 1.0, "0 dB, gain = 1", "#534AB7", "o"), (0.0, 2.0, "0 dB, gain = 2", "#A9A4DB", "o"),
              (5.0, 1.0, "+5 dB, gain = 1", "#D85A30", "s"), (5.0, 2.0, "+5 dB, gain = 2", "#EFA98F", "s"))


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
    rows = design_differences()
    if rows:
        design_figure(rows)
        print("\ndesign\n" + design_table(rows))


if __name__ == "__main__":
    main()

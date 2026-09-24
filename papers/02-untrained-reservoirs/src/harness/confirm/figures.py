"""The confirmatory figures, and the tables printed beside them, drawn from the record.

    uv run python -m harness.confirm.figures      # write the figures, print the paper's tables

Every number in a figure and in its table comes from the same accuracies that
`harness.confirm.summary` reports, so the paper cannot drift from the record.
"""

from __future__ import annotations

import argparse

from harness.confirm import score as sc
from harness.confirm import summary as sm
from harness.utils.paths import FIGURES_DIR

NOISES = ((None, "clean"), (0.0, "0 dB"), (5.0, "+5 dB"))

#: Tier 1 recognition, primary cell: (arm, read, gain, table label, legend label, colour)
RECOGNITION_ARMS = (
    ("floor", "windowed@wholeclip", None,
     "**Floor**: the front end alone, its 16 band envelopes read directly; no dynamics",
     "Floor: front end alone, no dynamics", "#8C8C8C"),
    (sm.FIELD, "windowed", 1.0,
     "**Oscillator field** (gain = 1): 1,024 coupled oscillators, untrained",
     "Oscillator field (gain = 1)", "#534AB7"),
    (sm.SEVERED, "windowed", 1.0,
     "**Severed field** (gain = 1): the same field with its coupling removed",
     "Severed field: coupling removed (gain = 1)", "#A9A4DB"),
    ("bank-c4", "windowed", 1.0,
     "**Bank A** (gain = 1): 1,024 leaky integrators with the field's states and parameters, untrained",
     "Bank A: leaky integrators, the field's size (gain = 1)", "#D85A30"),
    ("ann-transformer", "windowed", None,
     "**Transformer**: trained end to end, 1,968 parameters",
     "Transformer: trained end to end", "#0F6E56"),
)


def recognition_cells() -> dict[tuple, dict]:
    """(arm, read, gain, noise) -> the accuracy record at the primary cell."""
    cells = sc.load(["tier1-recognition-envelope"])
    return {(r["arm"], r["read"], r["gain"], r["noise"]): r for r in sm.accuracies(cells)
            if r["width"] == sc.PRIMARY_WIDTH and r["n_train"] == sc.PRIMARY_SIZE}


def recognition_table(acc: dict[tuple, dict]) -> str:
    lines = ["| arm | " + " | ".join(h for _, h in NOISES) + " |", "|---" * (len(NOISES) + 1) + "|"]
    for arm, read, gain, label, _, _ in RECOGNITION_ARMS:
        cells = [acc[(arm, read, gain, noise)] for noise, _ in NOISES]
        lines.append(f"| {label} | " + " | ".join(f"{c['mean']:.1f} ± {c['sd']:.1f}" for c in cells) + " |")
    return "\n".join(lines)


def recognition_figure(acc: dict[tuple, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    n, width = len(RECOGNITION_ARMS), 0.16
    for i, (arm, read, gain, _, legend, colour) in enumerate(RECOGNITION_ARMS):
        cells = [acc[(arm, read, gain, noise)] for noise, _ in NOISES]
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(NOISES))]
        means, sds = [c["mean"] for c in cells], [c["sd"] for c in cells]
        ax.bar(xs, means, width, yerr=sds, capsize=2.5, color=colour, label=legend,
               error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
        for x, m, s in zip(xs, means, sds, strict=True):
            ax.text(x, m + s + 1.2, f"{m:.1f}", ha="center", va="bottom", rotation=90, fontsize=7)
    ax.axhline(10, color="#333333", linewidth=0.8, linestyle=":")
    ax.text(len(NOISES) - 0.5, 11, "chance", ha="right", va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(range(len(NOISES)), [h for _, h in NOISES])
    ax.set_xlabel("noise level of the training and test audio")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    # the LaTeX builds take the vector PDF; everything else takes the PNG
    stem = FIGURES_DIR / "c1-recognition-accuracy"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 200})):
        fig.savefig(stem.with_suffix(suffix), **kwargs)
    plt.close(fig)
    print(f"wrote {stem}.{{pdf,png}}")


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    acc = recognition_cells()
    recognition_figure(acc)
    print(recognition_table(acc))


if __name__ == "__main__":
    main()

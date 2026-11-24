"""Model selection on the validation speakers, from the record.

Protocol V (harness.experiment.protocol) holds speakers 41-48 out of the
training pool. Every cell recorded under it carries `val_acc`, the ridge's
accuracy on those speakers at its chosen penalty. A configuration is selected
by that number alone, averaged over the seeds, and its test accuracy is read
off afterwards; the test set never enters the choice.

    select(cells, experiment, family)   the best configuration per noise level

The oscillator network's grid is the `selection` experiment (gain, restoring
strength and coupling ceiling for the Kuramoto and Stuart–Landau networks); the
reservoir baselines' grids are the `esn` and `ron` experiments. The summary
reports each selection, and `plan.selected` reruns the chosen reservoir
baselines under Protocol A, on clean audio and on the order task as well.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from harness.experiment.record import PRIMARY_SIZE, PRIMARY_WIDTH, Cell


def replicate(c: Cell) -> str:
    return f"fold{c.fold}" if c.fold >= 0 else f"seed{c.seed}"


def candidates(cells: list[Cell], experiment: str, family: str | None = None) -> list[Cell]:
    """The primary-width validation-protocol cells of one selection grid (and one coupling function, if given)."""
    return [c for c in cells if c.experiment == experiment and c.protocol == "V" and c.read == "windowed"
            and c.width == PRIMARY_WIDTH and c.n_train == PRIMARY_SIZE and c.val_acc is not None
            and (family is None or c.arm.get("coupling") == family)]


def by_configuration(cells: list[Cell]) -> dict[tuple[str, float | None], dict[str, Cell]]:
    """(label, gain) -> replicate -> cell, for cells of one noise level."""
    out: dict = defaultdict(dict)
    for c in cells:
        out[(c.label, c.gain)][replicate(c)] = c
    return out


def select(cells: list[Cell], experiment: str, family: str | None = None) -> dict[float | None, dict]:
    """Per noise level: the configuration with the highest mean validation accuracy over its replicates.

    Ties go to the alphabetically first label and the lower gain. Each entry
    gives the chosen label and gain, its replicates' cells, and the grid's size.
    """
    mine = candidates(cells, experiment, family)
    out = {}
    for noise in sorted({c.noise for c in mine}, key=lambda x: -1 if x is None else x):
        configs = by_configuration([c for c in mine if c.noise == noise])
        complete = {k: v for k, v in configs.items() if len(v) == max(len(r) for r in configs.values())}
        ranked = sorted(complete.items(),
                        key=lambda kv: (-statistics.fmean(c.val_acc for c in kv[1].values()), kv[0][0], kv[0][1]))
        (label, gain), reps = ranked[0]
        out[noise] = {"label": label, "gain": gain, "cells": reps, "grid": len(configs), "configs": configs}
    return out

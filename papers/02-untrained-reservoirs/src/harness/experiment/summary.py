"""Every accuracy and every comparison in the record, with its spread.

    uv run python -m harness.experiment.summary      # write summary.json and summary.md

No threshold decides anything here. An accuracy is reported as its mean over
replicates (three seeds under Protocol A, five folds under Protocol B), the
sample standard deviation and each replicate's own value, all in points. A
comparison A minus B is paired: matched on condition and replicate, it reports
each replicate's difference, their mean and standard deviation, and a 95%
interval from resampling test clips. Seeds share one test set, so their
per-clip differences are averaged; folds and order-task pairs each have their
own test clips, so theirs are concatenated. Whether a difference is a gain is
left to the reader.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

import numpy as np

from harness.experiment import plan, selection, terms
from harness.experiment import record as rec
from harness.experiment import run as rn
from harness.experiment.arms import Arm
from harness.experiment.record import Cell

COUPLED = rec.COUPLED_LABEL
UNCOUPLED = COUPLED.replace("coupled-", "uncoupled-", 1)
READ = {"recognition": "windowed", "order": "pooled"}
#: the reference level of each design factor: the reference network's own
REFERENCE = terms.REFERENCE
#: bootstrap resamples drawn per batch, bounding memory on Protocol B's 30,000 test clips
CHUNK = 100


def replicate(c: Cell) -> str:
    return f"fold{c.fold}" if c.fold >= 0 else f"seed{c.seed}"


def spread(values: dict[str, float]) -> dict:
    """Mean, sample standard deviation and each replicate's value, from fractions to points."""
    pts = {k: 100 * v for k, v in sorted(values.items())}
    xs = list(pts.values())
    return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else None,
            "n": len(xs), "values": pts}


def interval(per_clip: np.ndarray) -> list[float]:
    """The 95% interval of the mean from resampling clips, in points."""
    rng = np.random.default_rng(rec.BOOT_SEED)
    boots = np.concatenate([per_clip[rng.integers(0, len(per_clip), (CHUNK, len(per_clip)))].mean(axis=1)
                            for _ in range(rec.BOOTSTRAP // CHUNK)])
    lo, hi = np.quantile(boots, [(1 - rec.CONFIDENCE) / 2, (1 + rec.CONFIDENCE) / 2])
    return [100 * float(lo), 100 * float(hi)]


def paired(pairs: list[tuple[Cell, Cell]]) -> dict:
    """A minus B over matched cells.

    A replicate's difference is the mean over its own pairs (matched pairs,
    order-task digit pairs). The interval resamples test clips: per-clip differences are averaged
    over cells that share a test set and concatenated over test sets that do not.
    """
    by_rep = defaultdict(list)
    for a, b in pairs:
        by_rep[replicate(a)].append(a.acc - b.acc)
    out = {**spread({r: statistics.fmean(v) for r, v in by_rep.items()}), "n_pairs": len(pairs)}
    if all(a.bits and b.bits for a, b in pairs):
        by_set = defaultdict(list)
        for a, b in pairs:
            by_set[(a.fold, a.pair)].append(rec.bits(a) - rec.bits(b))
        out["ci95"] = interval(np.concatenate([np.mean(v, axis=0) for _, v in sorted(by_set.items())]))
    return out


# ---------------------------------------------------------------------------
# Accuracies
# ---------------------------------------------------------------------------

def accuracies(cells: list[Cell]) -> list[dict]:
    """One record per arm, read, condition, width and size, over its replicates."""
    groups = defaultdict(dict)
    for c in cells:
        key = (c.experiment, c.task, c.pathway, c.label, c.read, c.noise, c.gain, c.pair, c.width, c.n_train, c.projection,
               c.view)
        groups[key][replicate(c)] = c.acc
    fields = ("experiment", "task", "pathway", "arm", "read", "noise", "gain", "pair", "width", "n_train", "projection",
              "view")
    return [{**dict(zip(fields, k, strict=True)), "name": terms.arm(k[3]),
             "pair": list(k[7]) or None, **spread(v)} for k, v in sorted(groups.items(), key=str)]


def validation_accuracies(cells: list[Cell]) -> list[dict]:
    """Protocol V cells: the validation speakers' accuracy beside the test accuracy, per configuration."""
    groups = defaultdict(lambda: ({}, {}))
    for c in cells:
        if c.protocol == "V" and c.val_acc is not None:
            key = (c.experiment, c.task, c.label, c.read, c.noise, c.gain, c.width, c.n_train)
            groups[key][0][replicate(c)] = c.val_acc
            groups[key][1][replicate(c)] = c.acc
    fields = ("experiment", "task", "arm", "read", "noise", "gain", "width", "n_train")
    return [{**dict(zip(fields, k, strict=True)), "name": terms.arm(k[2]), "protocol": "V",
             "validation": spread(v), "test": spread(t)} for k, (v, t) in sorted(groups.items(), key=str)]


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def matched(a_cells: list[Cell], b_cells: list[Cell], widths: tuple | None = None) -> dict[tuple, list]:
    """Pairs (a, b) matched on condition, replicate, width and size, grouped by condition, width and size.

    `widths=(wa, wb)` takes a at width wa and b at width wb rather than width for
    width. An arm without gain is matched at every gain of the other.
    Order-task pairs are pooled into one group per condition.
    """
    b_gain = any(c.gain is not None for c in b_cells)
    idx = {(c.noise, c.gain, c.pair, c.width, c.n_train, replicate(c)): c for c in b_cells}
    groups = defaultdict(list)
    for c in a_cells:
        if widths and c.width != widths[0]:
            continue
        other = idx.get((c.noise, c.gain if b_gain else None, c.pair, widths[1] if widths else c.width,
                         c.n_train, replicate(c)))
        if other is not None:
            groups[(c.noise, c.gain, c.width, c.n_train)].append((c, other))
    return groups


def versus(cells: list[Cell], name: str, a: tuple[str, str], b: tuple[str, str], *, experiment: str, task: str,
           pathway: str = "spectrogram", widths: tuple | None = None, projection: str = "fixed") -> list[dict]:
    """a minus b, each an (arm, read), in one experiment and under one projection."""
    mine = [c for c in cells if c.experiment == experiment and c.task == task and c.pathway == pathway and c.projection == projection]
    groups = matched([c for c in mine if (c.label, c.read) == a], [c for c in mine if (c.label, c.read) == b], widths)
    return [{"comparison": name, "a": " ".join(a), "b": " ".join(b), "experiment": experiment, "task": task, "pathway": pathway,
             "noise": k[0], "gain": k[1], "width": k[2], "b_width": widths[1] if widths else k[2],
             "n_train": k[3], "projection": projection, **paired(v)} for k, v in sorted(groups.items(), key=str)]


BASELINE_WHOLE = "the spectrogram-only baseline, whole clip"
BASELINE_16 = "the spectrogram-only baseline, from frame 16"
NETWORK = terms.arm(COUPLED)


def _against_network(cells: list[Cell], experiment: str, task: str) -> list[dict]:
    r = READ[task]
    others = [(BASELINE_WHOLE, ("baseline", r + "@wholeclip")), (BASELINE_16, ("baseline", r)),
              ("the " + terms.arm(UNCOUPLED), (UNCOUPLED, r)), ("the " + terms.arm("bank-state"), ("bank-state", r)),
              ("the " + terms.arm("bank-width"), ("bank-width", r))]
    others += [(f"the {terms.arm('trained-' + arch)}", (f"trained-{arch}", r)) for arch in plan.TRAINED_ARCHS]
    out = []
    for name, b in others:
        out += versus(cells, f"{NETWORK} minus {name}", (COUPLED, r), b, experiment=experiment, task=task)
    return out


def controls(cells: list[Cell]) -> list[dict]:
    """The coupled network against every other arm, rotation rates, and width 4,096 against 192 per arm."""
    out = []
    for task in ("recognition", "order"):
        r = READ[task]
        out += _against_network(cells, "controls", task)
        out += versus(cells, f"{NETWORK}, with rotation rates minus without", (COUPLED, r + "+rate"), (COUPLED, r),
                      experiment="controls", task=task)
        for label in sorted({c.label for c in cells if c.experiment == "controls" and c.task == task}):
            read = r + "@wholeclip" if label == "baseline" else r
            name = terms.arm(label) + (", whole clip" if label == "baseline" else "")
            out += versus(cells, f"{name}: width 4,096 minus 192", (label, read), (label, read),
                          experiment="controls", task=task, widths=(4096, 192))
    return out


def design(cells: list[Cell]) -> list[dict]:
    """Each design factor level minus its reference, over matched pairs identical in every other factor."""
    prim = [c for c in cells if c.experiment == "design" and c.read == "windowed"]
    out = []
    for factor, ref in REFERENCE.items():
        def match(c: Cell, factor: str = factor) -> tuple:
            return (tuple(c.arm[f] for f in REFERENCE if f != factor), c.noise, c.gain, c.width, c.n_train,
                    replicate(c))
        refs = {match(c): c for c in prim if c.arm[factor] == ref}
        for level in sorted({c.arm[factor] for c in prim} - {ref}, key=str):
            groups = defaultdict(list)
            for c in prim:
                if c.arm[factor] == level and match(c) in refs:
                    groups[(c.noise, c.gain, c.width, c.n_train)].append((c, refs[match(c)]))
            name = f"{terms.FACTORS[factor]}: {terms.level(level)} minus {terms.level(ref)}"
            out += [{"comparison": name, "experiment": "design", "task": "recognition",
                     "pathway": "spectrogram", "noise": k[0], "gain": k[1], "width": k[2], "b_width": k[2],
                     "n_train": k[3], **paired(v)} for k, v in sorted(groups.items(), key=str)]
    # each coupling function at the reference configuration against the whole-clip baseline; the design
    # experiment's cells share the controls experiment's test clips and training draws, so they pair with its baseline
    base = [c for c in cells if c.experiment == "controls" and c.task == "recognition" and c.label == "baseline"
            and c.read == "windowed@wholeclip"]
    for coupling in plan.PHASE_COUPLINGS + plan.AMPLITUDE_COUPLINGS:
        label = Arm("network", coupling=coupling).label()
        groups = matched([c for c in prim if c.label == label], base)
        out += [{"comparison": f"{terms.arm(label)} minus {BASELINE_WHOLE}", "experiment": "design", "task": "recognition",
                 "pathway": "spectrogram", "noise": k[0], "gain": k[1], "width": k[2], "b_width": k[2],
                 "n_train": k[3], **paired(v)} for k, v in sorted(groups.items(), key=str)]
    # rotation rates, over every design configuration
    idx = {(c.label, c.noise, c.gain, c.width, c.n_train, replicate(c)): c for c in prim}
    groups = defaultdict(list)
    for c in cells:
        if c.experiment == "design" and c.read == "windowed+rate":
            other = idx.get((c.label, c.noise, c.gain, c.width, c.n_train, replicate(c)))
            if other is not None:
                groups[(c.noise, c.gain, c.width, c.n_train)].append((c, other))
    out += [{"comparison": "with rotation rates minus without, every design configuration", "experiment": "design",
             "task": "recognition",
             "pathway": "spectrogram", "noise": k[0], "gain": k[1], "width": k[2], "b_width": k[2], "n_train": k[3],
             **paired(v)} for k, v in sorted(groups.items(), key=str)]
    return out


def pathways(cells: list[Cell]) -> list[dict]:
    """Every arm on the quadrature pathway, minus that pathway's own baseline."""
    out = []
    for pathway in ("quadrature",):
        for label in sorted({c.label for c in cells if c.experiment == "quadrature" and c.pathway == pathway} - {"baseline"}):
            for name, read in ((BASELINE_WHOLE, "windowed@wholeclip"), (BASELINE_16, "windowed")):
                out += versus(cells, f"{terms.arm(label)} minus {name}", (label, "windowed"), ("baseline", read),
                              experiment="quadrature", task="recognition", pathway=pathway)
    # the same network on the quadrature pathway against the spectrogram pathway: the design experiment ran
    # every quadrature configuration with the same seeds, conditions and test clips
    spectrogram = [c for c in cells if c.experiment == "design" and c.read == "windowed"]
    quadrature = [c for c in cells if c.experiment == "quadrature" and c.pathway == "quadrature" and c.read == "windowed"]
    for label in sorted({c.label for c in quadrature} - {"baseline"}):
        groups = matched([c for c in quadrature if c.label == label], [c for c in spectrogram if c.label == label])
        out += [{"comparison": f"{terms.arm(label)}: quadrature minus spectrogram pathway", "experiment": "quadrature",
                 "task": "recognition", "pathway": "quadrature", "noise": k[0], "gain": k[1], "width": k[2],
                 "b_width": k[2], "n_train": k[3], **paired(v)} for k, v in sorted(groups.items(), key=str)]
    return out


def _pair_by(a_cells: list[Cell], b_cells: list[Cell], key) -> dict[tuple, list]:
    """Pairs (a, b) with equal key(a) == key(b), grouped by a's condition, width and size."""
    idx = {key(c): c for c in b_cells}
    groups = defaultdict(list)
    for c in a_cells:
        other = idx.get(key(c))
        if other is not None:
            groups[(c.noise, c.gain, c.width, c.n_train)].append((c, other))
    return groups


def _rows(name: str, experiment: str, groups: dict[tuple, list]) -> list[dict]:
    return [{"comparison": name, "experiment": experiment, "task": "recognition", "pathway": "spectrogram", "noise": k[0],
             "gain": k[1], "width": k[2], "b_width": k[2], "n_train": k[3], **paired(v)}
            for k, v in sorted(groups.items(), key=str)]


def sweep(cells: list[Cell]) -> list[dict]:
    """The sweep experiment against the design experiment's reference cells: each restoring strength minus 0.3, each ceiling minus
    1, and each gain minus gain 1, per coupling function, at the reference configuration."""
    design_cells = [c for c in cells if c.experiment == "design" and c.read == "windowed"]
    swept = [c for c in cells if c.experiment == "sweep" and c.read == "windowed"]
    out = []
    for coupling in plan.PHASE_COUPLINGS + plan.AMPLITUDE_COUPLINGS:
        reference = Arm("network", coupling=coupling)
        base = [c for c in design_cells if c.label == reference.label()]
        same = lambda c: (c.noise, c.gain, c.width, c.n_train, replicate(c))  # noqa: E731
        name = terms.level(coupling)
        for restoring in plan.SWEEP_RESTORINGS:
            a = [c for c in swept if c.label == Arm("network", coupling=coupling, restoring=restoring).label()]
            out += _rows(f"{name}: restoring strength {restoring:g} minus 0.3", "sweep", _pair_by(a, base, same))
        for ceiling in plan.SWEEP_CEILINGS:
            a = [c for c in swept if c.label == Arm("network", coupling=coupling, ceiling=ceiling).label()]
            out += _rows(f"{name}: coupling ceiling {ceiling:g} minus 1", "sweep", _pair_by(a, base, same))
        at_one = [c for c in base if c.gain == 1.0]
        by_seed = lambda c: (c.noise, c.width, c.n_train, replicate(c))  # noqa: E731
        for gain in plan.SWEEP_GAINS:
            a = [c for c in swept if c.label == reference.label() and c.gain == gain]
            out += _rows(f"{name}: gain {gain:g} minus gain 1", "sweep", _pair_by(a, at_one, by_seed))
    return out


def cochlea(cells: list[Cell]) -> list[dict]:
    """The coil and the cochlea against the design experiment's torus and helix, and the cochlea against the coil, over
    matched configurations (the same coupling function and natural frequencies, at the reference restoring
    strength and ceiling)."""
    runs = [c for c in cells if c.experiment in ("design", "cochlea") and c.read == "windowed"
            and c.arm["restoring"] == 0.3 and c.arm["ceiling"] == 1.0 and c.arm["coupling"] in plan.PHASE_COUPLINGS]

    def of(geometry):
        return [c for c in runs if c.arm["geometry"] == geometry]

    def key(c):
        return (c.arm["coupling"], c.arm["frequencies"], c.noise, c.gain, c.width, c.n_train, replicate(c))
    out = []
    for a, b in (("coil", "torus"), ("cochlea", "torus"), ("cochlea", "coil"), ("coil", "helix"),
                 ("cochlea-matched", "coil"), ("cochlea-matched", "torus"), ("cochlea", "cochlea-matched")):
        out += _rows(f"lattice geometry: {a} minus {b}", "cochlea", _pair_by(of(a), of(b), key))
    return out


def projections(cells: list[Cell]) -> list[dict]:
    """The projection experiment: each reservoir under the seeded projection minus the fixed one, and the coupled
    network against its controls under the seeded projection (the spectrogram-only baseline and the trained
    baselines are never projected, so their controls cells stand for both)."""
    out = []
    for task in ("recognition", "order"):
        r = READ[task]
        mine = [c for c in cells if c.experiment == "projection" and c.task == task and c.read == r]
        seeded = [c for c in mine if c.projection == "seeded"]
        for label in sorted({c.label for c in mine}):
            groups = matched([c for c in seeded if c.label == label],
                             [c for c in mine if c.label == label and c.projection == "fixed"])
            out += [{"comparison": f"{terms.arm(label)}: seeded minus fixed projection", "experiment": "projection",
                     "task": task, "noise": k[0], "gain": k[1], "width": k[2], "n_train": k[3],
                     **paired(v)} for k, v in sorted(groups.items(), key=str)]
        network = [c for c in seeded if c.label == COUPLED]
        controls = [(BASELINE_WHOLE, [c for c in cells if c.experiment == "controls" and c.task == task and c.label == "baseline"
                                      and c.read == r + "@wholeclip"])]
        controls += [("the " + terms.arm(lab), [c for c in seeded if c.label == lab]) for lab in (UNCOUPLED, "bank-state", "bank-width")]
        for name, b_cells in controls:
            groups = matched(network, b_cells)
            out += [{"comparison": f"{NETWORK} minus {name}, seeded projection", "experiment": "projection", "task": task,
                     "noise": k[0], "gain": k[1], "width": k[2], "n_train": k[3], "projection": "seeded",
                     **paired(v)} for k, v in sorted(groups.items(), key=str)]
    return out


def reproduction(cells: list[Cell]) -> dict:
    """The projection experiment's unseeded cells against the controls experiment's, cell for cell: the
    same runs, re-executed.

    The controls ran on the CPU and the projection experiment simulated its reservoirs on the Apple GPU, so this is
    how far the two devices' floating point moves a recorded cell. A cell counts as identical only if its
    per-clip correctness is identical too.
    """
    def key(c: Cell) -> tuple:
        return (c.task, c.label, c.read, c.width, c.n_train, c.noise, c.gain, c.seed, c.pair)
    reference = {key(c): c for c in cells if c.experiment == "controls"}
    pairs = [(c, reference[key(c)]) for c in cells
             if c.experiment == "projection" and c.projection != "seeded" and key(c) in reference]
    if not pairs:
        return {}
    clips = [round(abs(a.acc - b.acc) * a.n_test) for a, b in pairs]
    return {"cells": len(pairs),
            "identical": sum(a.acc == b.acc and a.bits == b.bits for a, b in pairs),
            "within_two_clips": sum(n <= 2 for n in clips),
            "largest_points": 100 * max(abs(a.acc - b.acc) for a, b in pairs),
            "largest_clips": max(clips)}


EXACT_DRIVE = {"stuart-landau-exact": "stuart-landau", "stuart-landau-fixed-exact": "stuart-landau-fixed"}
SL_LABEL = Arm("network", coupling="stuart-landau").label()


def _by_condition_and_seed(c: Cell) -> tuple:
    return (c.noise, c.gain, c.width, c.n_train, replicate(c))


def _against_gainless(a_cells: list[Cell], b_cells: list[Cell], name: str, experiment: str) -> list[dict]:
    """a minus a gainless b (the spectrogram-only baseline), matched at every gain of a."""
    return [{"comparison": name, "experiment": experiment, "task": "recognition", "pathway": "spectrogram",
             "noise": k[0], "gain": k[1], "width": k[2], "b_width": k[2], "n_train": k[3], **paired(v)}
            for k, v in sorted(matched(a_cells, b_cells).items(), key=str)]


def exact_drive(cells: list[Cell]) -> list[dict]:
    """The exact-drive experiment against the record: each exact-rotation Stuart–Landau network minus the same
    network with the semi-implicit drive (the design files at gains 1 and 2, the sweep files beyond), minus
    Kuramoto at the same gain, and gain g minus gain 1 within itself; the free-amplitude one minus the whole-clip
    baseline."""
    exact = [c for c in cells if c.experiment == "exact-drive" and c.read == "windowed"]
    record = [c for c in cells if c.experiment in ("design", "sweep") and c.read == "windowed"]
    kuramoto = [c for c in record if c.label == COUPLED]
    base = [c for c in cells if c.experiment == "controls" and c.task == "recognition" and c.label == "baseline"
            and c.read == "windowed@wholeclip"]
    out = []
    for coupling, semi in EXACT_DRIVE.items():
        a = [c for c in exact if c.label == Arm("network", coupling=coupling).label()]
        name = terms.level(coupling)
        b = [c for c in record if c.label == Arm("network", coupling=semi).label()]
        out += _rows(f"{name} minus {terms.level(semi)}", "exact-drive", _pair_by(a, b, _by_condition_and_seed))
        out += _rows(f"{name} minus Kuramoto", "exact-drive", _pair_by(a, kuramoto, _by_condition_and_seed))
        at_one = [c for c in a if c.gain == 1.0]
        by_seed = lambda c: (c.noise, c.width, c.n_train, replicate(c))  # noqa: E731
        for gain in sorted(plan.GAINS + plan.SWEEP_GAINS):
            if gain != 1.0:
                out += _rows(f"{name}: gain {gain:g} minus gain 1", "exact-drive",
                             _pair_by([c for c in a if c.gain == gain], at_one, by_seed))
        out += _against_gainless(a, base, f"{name} minus {BASELINE_WHOLE}", "exact-drive")
    return out


def clean(cells: list[Cell]) -> list[dict]:
    """The clean experiment against the controls experiment on clean audio: each Stuart–Landau network minus Kuramoto
    at the same gain, the free amplitude minus the fixed one, and each minus the whole-clip baseline."""
    mine = [c for c in cells if c.experiment == "clean" and c.read == "windowed"]
    if not mine:
        return []
    ctrl = [c for c in cells if c.experiment == "controls" and c.task == "recognition" and c.noise is None]
    kuramoto = [c for c in ctrl if c.label == COUPLED and c.read == "windowed"]
    base = [c for c in ctrl if c.label == "baseline" and c.read == "windowed@wholeclip"]
    free = [c for c in mine if c.label == Arm("network", coupling="stuart-landau").label()]
    fixed = [c for c in mine if c.label == Arm("network", coupling="stuart-landau-fixed").label()]
    out = []
    for coupling, a in (("stuart-landau", free), ("stuart-landau-fixed", fixed)):
        name = terms.level(coupling)
        out += _rows(f"{name} minus Kuramoto", "clean", _pair_by(a, kuramoto, _by_condition_and_seed))
        out += _against_gainless(a, base, f"{name} minus {BASELINE_WHOLE}", "clean")
    out += _rows(f"{terms.level('stuart-landau')} minus {terms.level('stuart-landau-fixed')}", "clean",
                 _pair_by(free, fixed, _by_condition_and_seed))
    return out


def _by_condition_seed_any_width(c: Cell) -> tuple:
    return (c.noise, c.gain, c.n_train, replicate(c))


ROW_HALVES = {"pooled-lower": "lower eight rows", "pooled-upper": "upper eight rows"}


def row_readout(cells: list[Cell]) -> list[dict]:
    """The row-readout experiment: each arm read through only the lower or only the upper eight rows of its lattice
    (the within-row pooling kept for one half, 96 features). The upper half minus the lower, each half minus the
    fixed random projection at the same width and minus the full pooled read at 192 (the projection-methods
    experiment; the whole-clip baseline's own 192 features for the baseline), and the Stuart–Landau network minus
    Kuramoto and each network minus the baseline through each half."""
    mine = [c for c in cells if c.experiment == "row-readout"]
    if not mine:
        return []
    full = [c for c in cells if c.experiment == "projection-methods" and c.read == "windowed" and c.projection == "pooled"]
    full += [c for c in cells if c.experiment == "controls" and c.task == "recognition" and c.label == "baseline"
             and c.read == "windowed@wholeclip" and c.width == rec.PRIMARY_WIDTH]
    out = []
    names = {COUPLED: NETWORK, UNCOUPLED: terms.arm(UNCOUPLED), SL_LABEL: terms.arm(SL_LABEL), "baseline": BASELINE_WHOLE}
    for label, name in names.items():
        arm = [c for c in mine if c.label == label]
        lower = [c for c in arm if c.projection == "pooled-lower"]
        upper = [c for c in arm if c.projection == "pooled-upper"]
        out += _rows(f"{name}: upper eight rows minus lower eight rows", "row-readout",
                     _pair_by(upper, lower, _by_condition_and_seed))
        for method, half in ROW_HALVES.items():
            a = [c for c in arm if c.projection == method]
            out += [{**r, "projection": method} for r in
                    _rows(f"{name}: {half} minus the random projection at width 96", "row-readout",
                          _pair_by(a, [c for c in arm if c.projection == "fixed"], _by_condition_and_seed))]
            out += [{**r, "projection": method} for r in
                    _rows(f"{name}: {half} minus every row (width 192)", "row-readout",
                          _pair_by(a, [c for c in full if c.label == label], _by_condition_seed_any_width))]
    for method, half in ROW_HALVES.items():
        m = [c for c in mine if c.projection == method]
        network = [c for c in m if c.label == COUPLED]
        sl = [c for c in m if c.label == SL_LABEL]
        base = [c for c in m if c.label == "baseline"]
        out += [{**r, "projection": method} for r in
                _rows(f"{terms.arm(SL_LABEL)} minus {NETWORK}, {half}", "row-readout",
                      _pair_by(sl, network, _by_condition_and_seed))]
        out += [{**r, "projection": method} for r in
                _rows(f"{NETWORK} minus the {terms.arm(UNCOUPLED)}, {half}", "row-readout",
                      _pair_by(network, [c for c in m if c.label == UNCOUPLED], _by_condition_and_seed))]
        for a, name in ((network, NETWORK), (sl, terms.arm(SL_LABEL))):
            out += [{**r, "projection": method} for r in
                    _against_gainless(a, base, f"{name} minus {BASELINE_WHOLE}, {half}", "row-readout")]
    return out


VIEW_NAMES = {"phase": "phase alone", "amplitude": "amplitude alone", "slaved": "slaved amplitude"}


def _by_condition_seed_projection(c: Cell) -> tuple:
    return (c.noise, c.gain, c.width, c.n_train, replicate(c), c.projection)


def band_readout(cells: list[Cell]) -> list[dict]:
    """The band-readout experiment: through each row alone, the Stuart–Landau network minus Kuramoto, the coupled
    network minus the uncoupled one, and each network minus the whole-clip baseline's own band."""
    mine = [c for c in cells if c.experiment == "band-readout" and c.projection.startswith("pooled-row")]
    if not mine:
        return []
    out = []
    network = [c for c in mine if c.label == COUPLED]
    sl = [c for c in mine if c.label == SL_LABEL]
    unc = [c for c in mine if c.label == UNCOUPLED]
    bank = [c for c in mine if c.label == "bank-state"]
    base = [c for c in mine if c.label == "baseline"]
    def tagged(rows, tag):
        return [{**r, "projection": tag, "band": int(tag[len("pooled-row"):])} for r in rows]
    for tag in sorted({c.projection for c in mine}):
        pick = lambda cs: [c for c in cs if c.projection == tag]  # noqa: E731
        out += tagged(_rows(f"{terms.arm(SL_LABEL)} minus {NETWORK}, one band", "band-readout",
                            _pair_by(pick(sl), pick(network), _by_condition_and_seed)), tag)
        out += tagged(_rows(f"{NETWORK} minus the {terms.arm(UNCOUPLED)}, one band", "band-readout",
                            _pair_by(pick(network), pick(unc), _by_condition_and_seed)), tag)
        out += tagged(_rows(f"{NETWORK} minus the {terms.arm('bank-state')}, one band", "band-readout",
                            _pair_by(pick(network), pick(bank), _by_condition_and_seed)), tag)
        for a, name in ((network, NETWORK), (sl, terms.arm(SL_LABEL)), (unc, terms.arm(UNCOUPLED)),
                        (bank, terms.arm("bank-state"))):
            out += tagged(_against_gainless(pick(a), pick(base), f"{name} minus {BASELINE_WHOLE}, one band",
                                            "band-readout"), tag)
    return out


def amplitude_views(cells: list[Cell]) -> list[dict]:
    """The amplitude-view experiment: the Stuart–Landau network through its phase alone and its amplitude alone
    against the Kuramoto network and against its full signals, and the Kuramoto network with a slaved amplitude
    against itself and against the Stuart–Landau network."""
    mine = [c for c in cells if c.experiment == "amplitude-view" and c.read == "windowed"]
    if not mine:
        return []
    kuramoto = [c for c in cells if c.experiment == "controls" and c.task == "recognition" and c.label == COUPLED
                and c.read == "windowed" and c.view == "full"]
    sl = {v: [c for c in mine if c.label == SL_LABEL and c.view == v] for v in ("full", "phase", "amplitude")}
    slaved = [c for c in mine if c.label == COUPLED and c.view == "slaved"]
    name = terms.arm(SL_LABEL)
    out = []
    for v in ("phase", "amplitude"):
        out += _rows(f"{name}, {VIEW_NAMES[v]} minus {NETWORK}", "amplitude-view",
                     _pair_by(sl[v], kuramoto, _by_condition_and_seed))
        out += _rows(f"{name} minus {name}, {VIEW_NAMES[v]}", "amplitude-view",
                     _pair_by(sl["full"], sl[v], _by_condition_and_seed))
    out += _rows(f"{name} minus {NETWORK} (this experiment's runs)", "amplitude-view",
                 _pair_by(sl["full"], kuramoto, _by_condition_and_seed))
    out += _rows(f"{NETWORK}, {VIEW_NAMES['slaved']} minus {NETWORK}", "amplitude-view",
                 _pair_by(slaved, kuramoto, _by_condition_and_seed))
    out += _rows(f"{NETWORK}, {VIEW_NAMES['slaved']} minus {name}", "amplitude-view",
                 _pair_by(slaved, sl["full"], _by_condition_and_seed))
    return out


def stiffness(cells: list[Cell]) -> list[dict]:
    """The stiffness experiment: the exact-drive Stuart–Landau network with a faster amplitude relaxation minus the
    exact-drive experiment's rate-1 network at the same gain, and each rate's gain 4 and 8 minus its gain 1."""
    mine = [c for c in cells if c.experiment == "stiffness" and c.read == "windowed"]
    if not mine:
        return []
    ref_label = Arm("network", coupling="stuart-landau-exact").label()
    ref = [c for c in cells if c.experiment == "exact-drive" and c.label == ref_label and c.read == "windowed"]
    out = []
    for label in sorted({c.label for c in mine}):
        a = [c for c in mine if c.label == label]
        out += _rows(f"{terms.arm(label)} minus {terms.arm(ref_label)}", "stiffness", _pair_by(a, ref, _by_condition_and_seed))
        at_one = {(c.noise, c.width, c.n_train, replicate(c)): c for c in a if c.gain == 1.0}
        groups = defaultdict(list)
        for c in a:
            if c.gain != 1.0 and (other := at_one.get((c.noise, c.width, c.n_train, replicate(c)))) is not None:
                groups[(c.noise, c.gain, c.width, c.n_train)].append((c, other))
        out += _rows(f"{terms.arm(label)}: gain minus gain 1", "stiffness", groups)
    at_one = {(c.noise, c.width, c.n_train, replicate(c)): c for c in ref if c.gain == 1.0}
    groups = defaultdict(list)
    for c in ref:
        if c.gain in (4.0, 8.0) and (other := at_one.get((c.noise, c.width, c.n_train, replicate(c)))) is not None:
            groups[(c.noise, c.gain, c.width, c.n_train)].append((c, other))
    out += _rows(f"{terms.arm(ref_label)}: gain minus gain 1", "stiffness", groups)
    return out


def selections(cells: list[Cell]) -> list[dict]:
    """Model selection on Protocol V's validation speakers: per network or reservoir baseline and noise level, the
    configuration with the highest mean validation accuracy, its test accuracy, and (for the networks) the
    reference configuration's test accuracy under the same protocol with the paired difference."""
    out = []
    for experiment, family in (("selection", "kuramoto"), ("selection", "stuart-landau"), ("esn", None), ("ron", None)):
        for noise, chosen in selection.select(cells, experiment, family).items():
            reps = chosen["cells"]
            entry = {"experiment": experiment, "family": family or experiment, "noise": noise,
                     "grid": chosen["grid"], "n_replicates": len(reps),
                     "selected": {"arm": chosen["label"], "name": terms.arm(chosen["label"]), "gain": chosen["gain"],
                                  "validation": spread({r: c.val_acc for r, c in reps.items()}),
                                  "test": spread({r: c.acc for r, c in reps.items()})}}
            if family is not None:
                reference = chosen["configs"].get((Arm("network", coupling=family).label(), 1.0))
                if reference:
                    entry["reference"] = {"arm": Arm("network", coupling=family).label(), "gain": 1.0,
                                          "validation": spread({r: c.val_acc for r, c in reference.items()}),
                                          "test": spread({r: c.acc for r, c in reference.items()})}
                    pairs = [(reps[r], reference[r]) for r in reps if r in reference]
                    entry["selected_minus_reference"] = paired(pairs)
            out.append(entry)
    return out


def calibration(cells: list[Cell]) -> list[dict]:
    """The calibration experiment's noise gate: per arm, gain and noise level, the validation and test accuracy on
    Protocol V, and whether the validation accuracy lands in the 70 to 90% window that fixed the study's noise levels.
    The baseline is read over the whole clip, as the paper reports it."""
    out = []
    for r in validation_accuracies(cells):
        if r["experiment"] != "calibration" or r["width"] != rec.PRIMARY_WIDTH or r["n_train"] != rec.PRIMARY_SIZE:
            continue
        if r["read"] != ("windowed@wholeclip" if r["arm"] == "baseline" else "windowed"):
            continue
        out.append({**r, "in_window": 70.0 <= r["validation"]["mean"] <= 90.0})
    return out


#: the selection grid's axes, each with the reference value the other two are held at in a profile
SELECTION_AXES = (("gain", 1.0), ("restoring", 0.3), ("ceiling", 1.0))
AXIS_NAMES = {"gain": "input gain", "restoring": "restoring strength", "ceiling": "coupling ceiling"}


def selection_profiles(cells: list[Cell]) -> list[dict]:
    """The selection grid read one axis at a time: per network and noise level, the validation and test accuracy
    along the gain, the restoring strength and the coupling ceiling with the other two factors at their reference
    values. It is the calibration the pilot ran, redone on the validation speakers."""
    out = []
    for family in ("kuramoto", "stuart-landau"):
        mine = selection.candidates(cells, "selection", family)
        for noise in sorted({c.noise for c in mine}, key=lambda x: -1 if x is None else x):
            configs = selection.by_configuration([c for c in mine if c.noise == noise])
            for axis, _ in SELECTION_AXES:
                fixed = {a: v for a, v in SELECTION_AXES if a != axis}
                points = []
                for (label, gain), reps in configs.items():
                    arm = next(iter(reps.values())).arm
                    values = {"gain": gain, "restoring": arm["restoring"], "ceiling": arm["ceiling"]}
                    if all(values[a] == v for a, v in fixed.items()):
                        points.append({"value": values[axis], "arm": label, "gain": gain, "n_replicates": len(reps),
                                       "validation": spread({r: c.val_acc for r, c in reps.items()}),
                                       "test": spread({r: c.acc for r, c in reps.items()})})
                if points:
                    out.append({"experiment": "selection", "family": family, "noise": noise, "axis": axis,
                                "fixed": fixed, "points": sorted(points, key=lambda p: p["value"])})
    return out


def reservoir_baselines(cells: list[Cell]) -> list[dict]:
    """The random nonlinear expansion against the coupled network (the controls experiment), the whole-clip
    baseline and the state-matched bank, on recognition and the order task; and the selected echo state network
    and random oscillators network (the baselines-selected experiment, Protocol A) against the same."""
    out = []
    for experiment in ("expansion", "baselines-selected"):
        for task in ("recognition", "order"):
            r = READ[task]
            mine = [c for c in cells if c.experiment == experiment and c.task == task and c.read == r]
            ctrl = [c for c in cells if c.experiment == "controls" and c.task == task]
            network = [c for c in ctrl if c.label == COUPLED and c.read == r]
            bank = [c for c in ctrl if c.label == "bank-state" and c.read == r]
            base = [c for c in ctrl if c.label == "baseline" and c.read == r + "@wholeclip"]
            key = lambda c: (c.noise, c.gain, c.pair, c.width, c.n_train, replicate(c))  # noqa: E731
            for label in sorted({c.label for c in mine}):
                a = [c for c in mine if c.label == label]
                name = terms.arm(label)
                for other, b in ((NETWORK, network), (terms.arm("bank-state"), bank)):
                    groups = _pair_by(b, a, key) if other == NETWORK else _pair_by(a, b, key)
                    comp = f"{NETWORK} minus the {name}" if other == NETWORK else f"{name} minus the {other}"
                    out += [{"comparison": comp, "experiment": experiment, "task": task, "pathway": "spectrogram",
                             "noise": k[0], "gain": k[1], "width": k[2], "b_width": k[2], "n_train": k[3], **paired(v)}
                            for k, v in sorted(groups.items(), key=str)]
                out += [{**row, "task": task} for row in _against_gainless(a, base, f"{name} minus {BASELINE_WHOLE}", experiment)]
    return out


def projection_methods(cells: list[Cell]) -> list[dict]:
    """The projection-methods experiment: each reservoir through a training-set PCA and through the within-row
    pooling minus the fixed random projection, and under each method the coupled network against its controls and
    the Stuart–Landau network against Kuramoto and against the whole-clip baseline."""
    mine = [c for c in cells if c.experiment == "projection-methods" and c.read == "windowed"]
    base = [c for c in cells if c.experiment == "controls" and c.task == "recognition" and c.label == "baseline"
            and c.read == "windowed@wholeclip"]
    fixed = [c for c in mine if c.projection == "fixed"]
    out = []
    for method in ("pca", "pooled"):
        m = [c for c in mine if c.projection == method]
        for label in sorted({c.label for c in m}):
            rows = _rows(f"{terms.arm(label)}: {method} minus fixed projection", "projection-methods",
                         _pair_by([c for c in m if c.label == label], [c for c in fixed if c.label == label],
                                  _by_condition_and_seed))
            out += [{**r, "projection": method} for r in rows]
        network = [c for c in m if c.label == COUPLED]
        sl = [c for c in m if c.label == SL_LABEL]
        for other in (UNCOUPLED, "bank-state", "bank-width"):
            rows = _rows(f"{NETWORK} minus the {terms.arm(other)}, {method} projection", "projection-methods",
                         _pair_by(network, [c for c in m if c.label == other], _by_condition_and_seed))
            out += [{**r, "projection": method} for r in rows]
        rows = _rows(f"{terms.arm(SL_LABEL)} minus {NETWORK}, {method} projection", "projection-methods",
                     _pair_by(sl, network, _by_condition_and_seed))
        out += [{**r, "projection": method} for r in rows]
        for a, name in ((network, NETWORK), (sl, terms.arm(SL_LABEL))):
            rows = _against_gainless(a, base, f"{name} minus {BASELINE_WHOLE}, {method} projection", "projection-methods")
            out += [{**r, "projection": method} for r in rows]
    return out


CHANCE = 0.1


def leak_check(cells: list[Cell]) -> dict:
    """With no input a reservoir's state carries nothing about the clip, so every cell of a zero-gain run
    must read exactly chance; anything else would mean the read or the data leaks around the reservoir."""
    zero = [c for c in cells if c.experiment == "leak-check" and c.gain == 0.0 and not c.label.endswith("@clipspan")]
    return {"runs": len({(c.label, c.seed) for c in zero}), "cells": len(zero),
            "off_chance": [f"{c.label} {c.read} width {c.width}: {100 * c.acc:.2f}%" for c in zero if c.acc != CHANCE]}


def summary(cells: list[Cell] | None = None) -> dict:
    cells = rec.load() if cells is None else cells
    return {"accuracy": accuracies(cells),
            "comparisons": (controls(cells) + _against_network(cells, "becker-folds", "recognition") + design(cells)
                            + pathways(cells) + projections(cells) + sweep(cells) + cochlea(cells)
                            + exact_drive(cells) + clean(cells) + reservoir_baselines(cells) + projection_methods(cells)
                            + row_readout(cells) + band_readout(cells) + amplitude_views(cells) + stiffness(cells)),
            "validation": validation_accuracies(cells),
            "selection": selections(cells),
            "calibration": calibration(cells),
            "selection_profiles": selection_profiles(cells),
            "order_baseline_at_chance": rec.baseline_at_chance(cells),
            "projection_reproduces_controls": reproduction(cells),
            "leak_check": leak_check(cells)}


# ---------------------------------------------------------------------------
# The readable report: the primary cells
# ---------------------------------------------------------------------------

def snr(n) -> str:
    """A noise level as reported: its signal-to-noise ratio. The record keeps the harness's
    level of the noise relative to the speech (protocol.add_noise), so its 5 is an SNR of -5 dB."""
    if n is None:
        return "clean"
    return "0 dB" if n == 0 else f"{-n:+g} dB".replace("-", "\u2212")



def _by_noise(r: dict) -> tuple:
    return (-1 if r["noise"] is None else r["noise"],), snr(r["noise"])


def _by_condition(r: dict) -> tuple:
    gain = f", gain = {r['gain']:g}" if r["gain"] is not None else ""
    return (-1 if r["noise"] is None else r["noise"], r["gain"] or 0), snr(r["noise"]) + gain


def _by_size(r: dict) -> tuple:
    return (r["n_train"], r["width"]), f"n {r['n_train']:,}, w {r['width']:,}"


def _acc(r: dict) -> str:
    return f"{r['mean']:.1f}" + (f" ± {r['sd']:.1f}" if r["sd"] is not None else "")


def _diff(r: dict) -> str:
    s = f"{r['mean']:+.2f}" + (f" ± {r['sd']:.2f}" if r["sd"] is not None else "")
    return s + (f" [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]" if "ci95" in r else "")


def _grid(records: list[dict], row, col, cell) -> list[str]:
    """A table: one row per `row(r)`, one column per `col(r)`, a (sort key, heading) pair."""
    rows: dict = defaultdict(dict)
    for r in records:
        rows[row(r)][col(r)] = cell(r)
    if not rows:
        return ["(not run yet)"]
    cols = sorted({c for v in rows.values() for c in v})
    out = ["| | " + " | ".join(h for _, h in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    return out + [f"| {k} | " + " | ".join(v.get(c, "") for c in cols) + " |" for k, v in rows.items()]


def _pooled(records: list[dict]) -> list[dict]:
    """Order-task accuracies averaged over pairs within each replicate."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        for rep, v in r["values"].items():
            groups[(r["arm"], r["read"], r["noise"], r["gain"])][rep].append(v / 100)
    return [{"arm": k[0], "read": k[1], "noise": k[2], "gain": k[3],
             **spread({rep: statistics.fmean(v) for rep, v in reps.items()})} for k, reps in groups.items()]


#: the order arms are listed in: the baseline, the reservoirs, then the trained baselines
ORDER = ["baseline", COUPLED, UNCOUPLED, "bank-state", "bank-width"] + [f"trained-{a}" for a in ("transformer", "gru", "s4d", "cnn", "tcn")]


def _rank(r: dict) -> tuple:
    return (ORDER.index(r["arm"]) if r["arm"] in ORDER else len(ORDER), not r["read"].endswith("@wholeclip"),
            r.get("gain") or 0)


def _pooled_projection(records: list[dict]) -> list[dict]:
    """Order-task accuracies averaged over pairs, kept apart by projection."""
    out = []
    for proj in ("fixed", "seeded"):
        out += [{**r, "projection": proj} for r in _pooled([r for r in records if r["projection"] == proj])]
    return out


def _arm(r: dict) -> str:
    name = terms.arm(r["arm"])
    if r["arm"] == "baseline":
        name += ", whole clip" if r["read"].endswith("@wholeclip") else ", from frame 16"
    if r.get("view", "full") != "full":
        name += f", {VIEW_NAMES[r['view']]}"
    return name + (f" (gain = {r['gain']:g})" if r.get("gain") is not None else "")


def progress() -> dict[str, tuple[int, int]]:
    """Runs recorded and runs planned, per experiment."""
    root, have, out = rn.record_root(), {}, {}
    for name, experiment in plan.EXPERIMENTS.items():
        specs = list(experiment())
        for s in specs:
            if s.group() not in have:
                path = root / f"{s.group()}.json"
                have[s.group()] = set(json.loads(path.read_text())["runs"]) if path.exists() else set()
        out[name] = (sum(s.run_id() in have[s.group()] for s in specs), len(specs))
    return out


def report(s: dict, done: dict[str, tuple[int, int]]) -> str:
    acc, cmp = s["accuracy"], s["comparisons"]
    w, n = rec.PRIMARY_WIDTH, rec.PRIMARY_SIZE

    def prim_acc(experiment, task, n_train=n):
        return sorted((r for r in acc if r["experiment"] == experiment and r["task"] == task and r["width"] == w
                       and r.get("projection", "fixed") == "fixed"
                       and r["n_train"] == n_train and r["read"].split("@")[0] == READ[task]), key=_rank)

    def prim_cmp(experiment, n_train=n, prefix=""):
        return [r for r in cmp if r["experiment"] == experiment and r["width"] == w and r["n_train"] == n_train
                and r["comparison"].startswith(prefix) and "width 4,096 minus" not in r["comparison"]]

    def by_gain(r):
        return r["comparison"] + (f" (gain = {r['gain']:g})" if r["gain"] is not None else "")

    rec_cell = (f"Readout width {w} and {n:,} training clips, the four-window read (frames 16 to 61); tested on the "
                "6,000 clips of speakers 49 to 60")
    ord_cell = (f"Readout width {w} and {n:,} training sequences per digit pair, the whole-span read (frames 16 to 147); "
                "tested on 2,048 sequences per pair from speakers 49 to 60")
    fit = (f"An arm with more than {w} signals is projected down to {w}; one with fewer is read as it is.")
    seeds = "mean ± standard deviation over seeds 0 to 2"
    diff = ("paired on the same test clips and seeds: mean ± standard deviation over seeds 0 to 2, with the 95% "
            "interval from resampling test clips in brackets, in points")
    lk = s["leak_check"]
    lines = ["# Results", "",
             "Generated by `uv run python -m harness.experiment.summary` from the record in this folder; every number "
             "here and more (every width, size, pair and read) is in `summary.json`, and `README.md` says what each "
             "record file holds. Accuracies are mean ± standard deviation over replicates (three seeds; five folds "
             "on Becker et al.'s folds), in percent. Differences are paired, mean ± standard deviation over "
             "replicates, with the 95% interval from resampling test clips in brackets, in points. Noise levels are "
             f"signal-to-noise ratios. Primary cell: width {w}, {n:,} training clips.", "",
             "Runs recorded: " + "; ".join(f"{k} {d:,} of {p:,}" for k, (d, p) in done.items()) + ".", "",
             ("Leak check: not run yet." if not lk["cells"] else
              f"Leak check: with no input, {lk['cells'] - len(lk['off_chance'])} of the {lk['cells']} cells of the "
              f"{lk['runs']} zero-gain runs read exactly chance (10%)"
              + (f"; off chance: {'; '.join(lk['off_chance'])}." if lk["off_chance"] else ".")), ""]
    lines += ["## Controls, recognition: accuracy (Section 4.1)", "",
              f"{rec_cell}; {seeds}, in percent. The spectrogram-only baseline is read both over the whole clip and, "
              f"as every other arm is, from frame 16. {fit}", ""]
    lines += _grid(prim_acc("controls", "recognition"), _arm, _by_noise, _acc)
    lines += ["", "## Controls, recognition: the coupled network minus each other arm (Section 4.1)", "",
              f"The cells of the table above, {diff}. The gain in brackets is the coupled network's, and the other arm's when it takes one.", ""]
    lines += _grid([r for r in prim_cmp("controls") if r["task"] == "recognition"], by_gain,
                   _by_noise, _diff)
    lines += ["", "## Controls, recognition: accuracy by readout width and training size (Section 4.7)", "",
              "Each column is a training size (n clips, nested: each set contains the smaller) and a readout width (w); "
              f"the four-window read, tested on the 6,000 clips of speakers 49 to 60; {seeds}, in percent. At each width "
              "an arm with more signals is projected down to it, and one with fewer is read as it is.", ""]
    for noise in (0.0, 5.0):
        rows = [r for r in acc if r["experiment"] == "controls" and r["task"] == "recognition" and r["noise"] == noise
                and r["read"] in ("windowed", "windowed@wholeclip") and r["width"] != "native"]
        rows.sort(key=_rank)
        lines += [f"### {snr(noise)}", ""]
        lines += _grid(rows, _arm, _by_size, _acc) + [""]
    lines += ["## Controls, recognition: readout width 4,096 minus 192 (Section 4.7)", "",
              f"Each arm at width 4,096 minus itself at {w}, both at {n:,} training clips, {diff}.", ""]
    lines += _grid([r for r in cmp if r["experiment"] == "controls" and r["task"] == "recognition" and r["n_train"] == n
                    and "width 4,096 minus" in r["comparison"]], by_gain, _by_noise, _diff)
    lines += ["", "## Controls, order task: accuracy, averaged over the five digit pairs (Section 4.2)", "",
              f"{ord_cell}; each seed's accuracy averaged over the five pairs, then {seeds}, in percent. "
              f"{fit} Chance is 50%.", ""]
    lines += _grid(sorted(_pooled(prim_acc("controls", "order")), key=_rank), _arm, _by_noise, _acc)
    lines += ["", "## Controls, order task: the coupled network minus each other arm, pooled over the five pairs (Section 4.2)", "",
              f"The cells of the table above, {diff}; the pairs' test clips are pooled.", ""]
    lines += _grid([r for r in prim_cmp("controls") if r["task"] == "order"], by_gain,
                   _by_noise, _diff)
    at = s["order_baseline_at_chance"]
    off = [k for k, v in at.items() if not all(r["contains_chance"] for r in v.values())]
    lines += ["", "Order task, the spectrogram-only baseline's 95% interval contains chance (50%) in "
              f"{len(at) - len(off)} of {len(at)} pair and noise cells"
              + (f"; it does not in: {', '.join(off)}." if off else "."), ""]
    lines += ["## Design: each level minus its reference, over matched configurations (Sections 4.3 to 4.5)", "",
              f"{rec_cell}. Each level minus its reference over every configuration identical in the other factors, "
              f"{diff}.", ""]
    lines += _grid(prim_cmp("design"), lambda r: r["comparison"], _by_condition, _diff)
    proj = [r for r in acc if r["experiment"] == "projection" and r["width"] == w and r["n_train"] == n]
    lines += ["", "## Projection: the controls experiment's reservoirs under the fixed and a seeded projection (Section 4.7)", "",
              "What the projection does: every arm's standardized summary statistics are multiplied by a random "
              "Gaussian matrix that maps them to the common readout width, so an arm with thousands of signals and one "
              "with 192 are read by readouts of the same size. The fixed projection is one draw, shared by every run "
              "and seed; the seeded projection is drawn afresh from each run's seed. Seeded minus fixed therefore "
              "measures how far the particular draw moves a result, which the spread over seeds otherwise leaves out.", "",
              f"Readout width {w} and {n:,} training clips or sequences per pair, each reservoir read through the fixed "
              f"projection and through one drawn from the run's seed; recognition with the four-window read, the order "
              f"task with the whole-span read, averaged over the five pairs; {seeds}, in percent.", ""]
    if proj:
        mine = [r for r in cmp if r["experiment"] == "projection" and r["width"] == w and r["n_train"] == n]
        moved = []
        for task in ("recognition", "order"):
            rows = [r for r in mine if r["task"] == task and r["comparison"].endswith("seeded minus fixed projection")]
            inside = sum("ci95" in r and r["ci95"][0] <= 0 <= r["ci95"][1] for r in rows)
            moved.append(f"by {min(r['mean'] for r in rows):+.2f} to {max(r['mean'] for r in rows):+.2f} points on "
                         f"{'the order task' if task == 'order' else task} (the 95% interval contains zero in {inside} "
                         f"of its {len(rows)} comparisons)")
        fixed = {(r["task"], r["comparison"], r["noise"], r["gain"]): r["mean"] for r in cmp
                 if r["experiment"] == "controls" and r["width"] == w and r["n_train"] == n}
        signs = [(r["mean"], fixed[k]) for r in mine if r["comparison"].endswith(", seeded projection")
                 and (k := (r["task"], r["comparison"].removesuffix(", seeded projection"), r["noise"], r["gain"])) in fixed]
        kept = sum((a > 0) == (b > 0) for a, b in signs)
        lines += [f"Seeded minus fixed, the reservoirs moved {' and '.join(moved)}. Under the seeded projection, "
                  + ("every one" if kept == len(signs) else f"{kept}") + f" of the {len(signs)} differences between the "
                  "coupled network and its controls keeps the sign it has under the fixed one.", ""]
        rp = s["projection_reproduces_controls"]
        lines += [f"Rerun on the Apple GPU, the projection experiment's fixed-projection cells against the controls "
                  f"experiment's on the CPU: {rp['identical']:,} of {rp['cells']:,} identical, {rp['within_two_clips']:,} "
                  f"within two test clips, largest difference {rp['largest_points']:.2f} points ({rp['largest_clips']} clips).", ""]
    for task in ("recognition", "order"):
        rows = [r for r in proj if r["task"] == task]
        rows = _pooled_projection(rows) if task == "order" else rows
        lines += [f"### {task}", ""]
        lines += _grid(sorted(rows, key=_rank), lambda r: f"{_arm(r)}, {r['projection']}", _by_noise, _acc) + [""]
    lines += _grid([r for r in cmp if r["experiment"] == "projection" and r["width"] == w and r["n_train"] == n],
                   lambda r: f"{r['task']}: {r['comparison']}" + (f" (gain = {r['gain']:g})" if r["gain"] is not None else ""),
                   _by_noise, _diff)
    lines += ["", "## Sweep: restoring strength, coupling ceiling and gain beyond the design experiment's, each minus its reference (Section 4.5)", "",
              f"{rec_cell}. Each coupling function at the reference configuration, {diff}.", ""]
    lines += _grid([r for r in cmp if r["experiment"] == "sweep" and r["width"] == w and r["n_train"] == n],
                   lambda r: r["comparison"], _by_condition, _diff)
    lines += ["", "## Exact-rotation drive: the Stuart–Landau networks with the input applied as an exact rotation (Section 4.5, Appendix E)", "",
              f"{rec_cell}. The record's Stuart–Landau runs apply the input inside a semi-implicit step that rotates the "
              "state by 2·arcsin(h/2) per frame rather than h (h = dt × gain × input), within 0.3% at gains 1 and 2 but "
              "7% too far by gain 8; these runs apply it as an exact rotation. Each exact-drive network minus the same "
              "network with the semi-implicit drive, minus Kuramoto at the same gain, minus itself at gain 1, and the "
              f"free-amplitude one minus the whole-clip baseline; {diff}.", ""]
    lines += _grid([r for r in cmp if r["experiment"] == "exact-drive" and r["width"] == w and r["n_train"] == n],
                   lambda r: r["comparison"], _by_condition, _diff)
    exact_acc = [r for r in acc if r["experiment"] == "exact-drive" and r["width"] == w and r["n_train"] == n
                 and r["read"] == "windowed"]
    lines += ["", "Accuracy of each exact-drive network by gain:", ""]
    lines += _grid(exact_acc, lambda r: f"{r['name']}, {snr(r['noise'])}", lambda r: (r["gain"], f"gain {r['gain']:g}"), _acc)
    clean_cmp = [r for r in cmp if r["experiment"] == "clean" and r["width"] == w and r["n_train"] == n]
    if clean_cmp:
        lines += ["", "## Clean audio: the Stuart–Landau networks beside Kuramoto and the input (Section 4.3, Appendix D.2)", "",
                  f"{rec_cell}, on clean audio, which the design experiment left out (the controls experiment ran the free amplitude clean; the fixed amplitude is new). Each Stuart–Landau network minus the "
                  f"controls experiment's Kuramoto network at the same gain, the free amplitude minus the fixed one, and each "
                  f"minus the whole-clip baseline; {diff}.", ""]
        lines += _grid(clean_cmp, lambda r: r["comparison"], _by_condition, _diff)
        clean_acc = [r for r in acc if r["experiment"] == "clean" and r["width"] == w and r["n_train"] == n
                     and r["read"] == "windowed"]
        lines += ["", "Accuracy of each Stuart–Landau network on clean audio by gain:", ""]
        lines += _grid(clean_acc, lambda r: r["name"], lambda r: (r["gain"], f"gain {r['gain']:g}"), _acc)
    row_cmp = [r for r in cmp if r["experiment"] == "row-readout" and r["n_train"] == n]
    if row_cmp:
        rw = plan.ROW_READOUT_WIDTH
        lines += ["", "## Row-restricted readouts: each arm through only the lower or only the upper eight rows of its lattice (Section 5, Appendix D.2)", "",
                  f"{n:,} training clips, the four-window read (frames 16 to 61); tested on the 6,000 clips of speakers 49 "
                  "to 60. The Kuramoto, uncoupled and Stuart–Landau networks and the whole-clip spectrogram baseline, each read through "
                  "the within-row pooling kept for one half of the rows only (8 rows × 3 statistics × 4 windows = "
                  f"{rw} features), beside the fixed random projection of every feature at width {rw}. The lower eight rows "
                  "are the eight lowest mel bands and the upper eight the highest; the baseline's rows are its bands. "
                  f"{seeds}, in percent.", ""]
        row_acc = [r for r in acc if r["experiment"] == "row-readout" and r["n_train"] == n]
        proj_name = {"fixed": f"random projection, width {rw}", **ROW_HALVES}
        lines += _grid(sorted(row_acc, key=_rank), lambda r: f"{_arm(r)}, {proj_name.get(r['projection'], r['projection'])}",
                       _by_noise, _acc)
        lines += ["", f"The upper half minus the lower, each half minus the random projection at width {rw} and minus the "
                  "full pooled read of every row (width 192; the projection-methods experiment, and the controls "
                  f"experiment's whole-clip baseline), and the study's comparisons through each half; {diff}.", ""]
        lines += _grid(row_cmp, lambda r: r["comparison"] + (f" (gain = {r['gain']:g})" if r["gain"] is not None else ""),
                       _by_noise, _diff)
    band_cmp = [r for r in cmp if r["experiment"] == "band-readout" and r["n_train"] == n]
    if band_cmp:
        bw = plan.BAND_READOUT_WIDTH
        lines += ["", "## Per-band readout accuracy: each arm through one row of its lattice alone (Appendix D.2)", "",
                  f"{n:,} training clips, the four-window read (frames 16 to 61); tested on the 6,000 clips of speakers 49 "
                  "to 60. The Kuramoto, uncoupled and Stuart–Landau networks, the state-matched bank and the whole-clip "
                  "spectrogram baseline, each read through the within-row pooling kept for one row (3 statistics × 4 "
                  f"windows = {bw} features), for every row; the baseline's rows are its bands. Reservoirs at gain 1. "
                  f"{seeds}, in percent; chance is 10%.", ""]
        band_acc = [r for r in acc if r["experiment"] == "band-readout" and r["n_train"] == n
                    and r["projection"].startswith("pooled-row")]
        band_of = lambda r: int(r["projection"][len("pooled-row"):])  # noqa: E731
        for noise in sorted({r["noise"] for r in band_acc}, key=lambda x: (x is not None, x)):
            lines += ["", f"{snr(noise)}:", ""]
            lines += _grid([r for r in band_acc if r["noise"] == noise], lambda r: _arm(r).replace(" (gain = 1)", ""),
                           lambda r: (band_of(r), f"band {band_of(r)}"), lambda r: f"{r['mean']:.1f}")
        lines += ["", f"The study's comparisons through one band at a time; {diff}; the interval is left out for room.", ""]
        for noise in sorted({r["noise"] for r in band_cmp}, key=lambda x: (x is not None, x)):
            lines += ["", f"{snr(noise)}:", ""]
            lines += _grid([r for r in band_cmp if r["noise"] == noise], lambda r: r["comparison"],
                           lambda r: (r["band"], f"band {r['band']}"), lambda r: f"{r['mean']:+.1f} ± {r['sd']:.1f}")
    view_cmp = [r for r in cmp if r["experiment"] == "amplitude-view" and r["width"] == w and r["n_train"] == n]
    if view_cmp:
        lines += ["", "## Amplitude views: what the readout sees of the Stuart–Landau network's z (Section 5, Appendix D.2)", "",
                  f"{rec_cell}. The Stuart–Landau network read through its full signals (r·sin θ, r·cos θ), through its "
                  "phase alone (every oscillator at unit radius) and through its amplitude alone (|z|, one signal per "
                  "oscillator), and the Kuramoto network read with an amplitude slaved to its phases (the Stuart–Landau "
                  "radial equation driven by the Kuramoto phases, with no feedback on them). Accuracy, "
                  f"{seeds}, in percent.", ""]
        view_acc = [r for r in acc if r["experiment"] == "amplitude-view" and r["width"] == w and r["n_train"] == n
                    and r["read"] == "windowed"]
        lines += _grid(view_acc, _arm, _by_condition, _acc)
        lines += ["", f"Each view against the controls experiment's Kuramoto network and the full read; {diff}.", ""]
        lines += _grid(view_cmp, lambda r: r["comparison"], _by_condition, _diff)
    stiff_cmp = [r for r in cmp if r["experiment"] == "stiffness" and r["width"] == w and r["n_train"] == n]
    if stiff_cmp:
        lines += ["", "## Amplitude relaxation rate: the exact-drive Stuart–Landau network with a faster amplitude (Section 5, Appendix D.2)", "",
                  f"{rec_cell}, 0 dB. The amplitude's relaxation rate alpha (and beta, so the attracting radius stays 1) "
                  "at 4 and 16 times the core's 0.05, a time constant of about 5 and 1.25 frames against 20, at gains "
                  f"1, 4 and 8; the exact-drive experiment's rate-1 network beside them. {seeds}, in percent.", ""]
        ref_label = Arm("network", coupling="stuart-landau-exact").label()
        stiff_acc = [r for r in acc if r["n_train"] == n and r["width"] == w and r["read"] == "windowed" and r["noise"] == 0.0
                     and (r["experiment"] == "stiffness" or (r["experiment"] == "exact-drive" and r["arm"] == ref_label
                                                             and r["gain"] in plan.STIFFNESS_GAINS))]
        lines += _grid(stiff_acc, lambda r: terms.arm(r["arm"]), lambda r: (r["gain"], f"gain {r['gain']:g}"), _acc)
        lines += ["", f"Each faster rate minus rate 1 at the same gain, and each rate's gain 4 and 8 minus its gain 1; {diff}.", ""]
        lines += _grid(stiff_cmp, lambda r: r["comparison"], lambda r: (r["gain"], f"gain {r['gain']:g}"), _diff)
    family = lambda f: terms.level(f) if f in terms.LEVELS else f  # noqa: E731
    lines += ["", "## Calibration on the validation speakers (Protocol V; Appendix F)", "",
              "The pilot's calibration redone in this repository under the study's own readout, on the validation speakers "
              "(41 to 48) and never the test speakers. The noise gate: the spectrogram-only baseline (whole clip) and the "
              "Kuramoto and Stuart–Landau networks at the reference configuration, from noise 10 dB below the speech to "
              f"10 dB above it; validation accuracy, {seeds}, in percent. The study's noise levels are those at which the "
              "strongest untrained network lands between 70 and 90%; a cell outside that window says so.", ""]
    lines += _grid(s["calibration"], lambda r: r["name"] + (f", gain = {r['gain']:g}" if r["gain"] is not None else ""),
                   _by_noise, lambda r: _acc(r["validation"]) + ("" if r["in_window"] else " (outside)"))
    lines += ["", "The selection grid one axis at a time, the other two factors at their reference values (gain 1, restoring "
              "strength 0.3, coupling ceiling 1): validation accuracy, with the test accuracy in brackets.", ""]
    for axis, name in AXIS_NAMES.items():
        flat = [{"row": f"{family(p['family'])}, {snr(p['noise'])}", **pt}
                for p in s["selection_profiles"] if p["axis"] == axis for pt in p["points"]]
        lines += ["", f"{name}:", ""] + _grid(flat, lambda r: r["row"], lambda r: (r["value"], f"{r['value']:g}"),
                                              lambda r: f"{_acc(r['validation'])} [{_acc(r['test'])}]")
    sel = s["selection"]
    lines += ["", "## Model selection on the validation speakers (Protocol V; Section 4.8, Appendix F)", "",
              "Protocol V trains on speakers 1 to 40, chooses the ridge penalty and selects the configuration on "
              "speakers 41 to 48 (4,000 clips), and tests on speakers 49 to 60 as every other experiment does. For each "
              "network and reservoir baseline, the configuration with the highest validation accuracy (mean over seeds) "
              f"is selected, and its test accuracy read off afterwards; {seeds}, in percent. For the networks the "
              "reference configuration (gain 1, restoring strength 0.3, coupling ceiling 1) under the same protocol "
              "stands beside it, with the paired difference.", ""]
    if sel:
        lines += ["| network or baseline | SNR | grid | selected configuration | validation | test | reference, test | selected minus reference |",
                  "|---|---|---|---|---|---|---|---|"]
        for e in sel:
            cfg = e["selected"]["name"] + (f" (gain = {e['selected']['gain']:g})" if e["selected"]["gain"] is not None else "")
            ref = _acc(e["reference"]["test"]) if "reference" in e else ""
            d = _diff(e["selected_minus_reference"]) if "selected_minus_reference" in e else ""
            lines += [f"| {terms.level(e['family']) if e['family'] in terms.LEVELS else e['family']} | {snr(e['noise'])} | "
                      f"{e['grid']} | {cfg} | {_acc(e['selected']['validation'])} | {_acc(e['selected']['test'])} | {ref} | {d} |"]
    else:
        lines += ["(not run yet)"]
    lines += ["", "## Reservoir baselines: the random nonlinear expansion and the selected echo state and random oscillators networks (Section 4.8)", "",
              f"{rec_cell}; the order task as in Section 4.2. The expansion is tanh(V·(gain × input) + b), a memoryless "
              "random feature map of each frame, at the network's 1,024 states and at its 2,048 signals; the echo state "
              "network and the random oscillators network run at their validation-selected hyperparameters (the table "
              f"above) under Protocol A. {seeds}, in percent.", ""]
    for task in ("recognition", "order"):
        rows = [r for r in acc if r["experiment"] in ("expansion", "baselines-selected") and r["task"] == task
                and r["width"] == w and r["n_train"] == n and r["read"] == READ[task]]
        rows = _pooled(rows) if task == "order" else rows
        lines += [f"### {task}", ""] + _grid(rows, _arm, _by_noise, _acc) + [""]
    lines += [f"Each against the coupled network, the whole-clip baseline and the state-matched bank, {diff}.", ""]
    lines += _grid([r for r in cmp if r["experiment"] in ("expansion", "baselines-selected") and r["width"] == w and r["n_train"] == n],
                   lambda r: f"{r['task']}: {r['comparison']}" + (f" (gain = {r['gain']:g})" if r["gain"] is not None else ""),
                   _by_noise, _diff)
    lines += ["", "## Projection methods: a training-set PCA and a within-row pooling beside the random projection (Section 3.3, Appendix D)", "",
              f"{rec_cell}. The same reservoir runs read three ways at width {w}: through the fixed random projection, "
              "through the leading 192 principal directions of the training clips' standardized features (fitted on "
              "the training clips alone), and through a structured pooling that averages each window's statistic over "
              "the signals of each lattice row (channels, columns, sin and cos), which gives 16 rows × 3 statistics × 4 "
              f"windows = 192, the spectrogram-only baseline's own layout. {seeds}, in percent.", ""]
    pm = [r for r in acc if r["experiment"] == "projection-methods" and r["width"] == w and r["n_train"] == n]
    lines += _grid(sorted(pm, key=_rank), lambda r: f"{_arm(r)}, {r['projection']}", _by_noise, _acc)
    lines += ["", f"Each method minus the fixed projection per arm, and the study's comparisons under each method, {diff}.", ""]
    lines += _grid([r for r in cmp if r["experiment"] == "projection-methods" and r["width"] == w and r["n_train"] == n],
                   lambda r: r["comparison"] + (f" (gain = {r['gain']:g})" if r["gain"] is not None else ""),
                   _by_noise, _diff)
    lines += ["", "## Cochlea: the coil and the cochlea, each minus another geometry, over matched configurations (Section 4.4)", "",
              f"{rec_cell}. Matched on coupling function and natural frequencies, {diff}.", ""]
    lines += _grid([r for r in cmp if r["experiment"] == "cochlea" and r["width"] == w and r["n_train"] == n],
                   lambda r: r["comparison"], _by_condition, _diff)
    lines += ["", "## Becker folds: accuracy on Becker et al.'s five speaker folds, clean (Section 4.1)", "",
              f"Readout width {w} and {plan.BECKER_TRAIN:,} training clips from 36 speakers per fold, the ridge penalty "
              "chosen on the fold's 12 validation speakers, the four-window read; tested on the fold's 6,000 clips of "
              f"12 other speakers; mean ± standard deviation over the five folds, in percent. {fit}", ""]
    lines += _grid(prim_acc("becker-folds", "recognition", plan.BECKER_TRAIN), _arm, _by_noise, _acc)
    lines += ["", "## Becker folds: the coupled network minus each other arm", "",
              "The cells of the table above, paired on the same test clips and folds: mean ± standard deviation over "
              "the five folds, with the 95% interval from resampling test clips in brackets, in points.", ""]
    lines += _grid(prim_cmp("becker-folds", plan.BECKER_TRAIN), by_gain, _by_noise, _diff)
    lines += ["", "## Quadrature: each network minus the quadrature front end's own baseline, and minus the same network on the spectrogram pathway (Section 4.6)", "",
              f"{rec_cell}. {diff[0].upper() + diff[1:]}.", ""]
    lines += _grid([r for r in prim_cmp("quadrature")], lambda r: f"{terms.PATHWAYS[r['pathway']]} pathway: {r['comparison']}",
                   _by_condition, _diff)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    s = summary()
    root = rn.record_root()
    (root / "summary.json").write_text(json.dumps(s, indent=1, default=str) + "\n")
    text = report(s, progress())
    (root / "summary.md").write_text(text)
    print(text)
    print(f"wrote {root / 'summary.json'} and {root / 'summary.md'}")


if __name__ == "__main__":
    main()

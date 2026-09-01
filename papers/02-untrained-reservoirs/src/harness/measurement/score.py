"""Score the results tree against the paper's pre-registered bars.

One entry point for every verdict in the paper:

    uv run python -m harness.measurement.score baselines    # floors, conventional refs, ladder
    uv run python -m harness.measurement.score envelope     # the primary drive
    uv run python -m harness.measurement.score quadrature
    uv run python -m harness.measurement.score carrier
    uv run python -m harness.measurement.score all --figures

Everything is read from `results/`; nothing is recomputed, so scoring is fast
and works on a checkout with no corpus present. The one exception is the
no-dynamics floor: it needs the stimulus bank, so it is computed once by the
baselines script and cached to `results/baselines/floors.json`. Without that
file the reports still print, with floor-relative columns marked unavailable.

Every threshold below was registered before the runs it scores; they are named
in the printout so a reader can check the verdict against the bar as written.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

from harness.results import DRIVES, Run, iter_runs, results_root
from harness.utils.paths import FIGURES_DIR

# --- pre-registered decision bars ------------------------------------------
BAR_GEOMETRY = 3.0        # points a shape must beat its torus twin by
BAR_DESIGNED_OMEGA = 5.0  # points designed omega must beat randomized omega by
BAR_TRANSDUCTION = 5.0    # points a drive must add over its own floor to claim restoration
BAR_FAMILY_SPREAD = 3.0   # points a physics family must move to count as a factor
BAR_COHERENCE_RHO = -0.3  # Spearman rho for the anti-correlation claim
BAR_ORDER_ACC = 0.60      # order-task accuracy on at least 3 of 5 pairs
GEOMETRIES = ("torus", "cylinder", "sheet", "helix", "cube", "sphere")
FAMILIES = ("kuramoto", "sakaguchi", "harmonic2", "winfree", "sl", "sl-fixedamp")
PRIMARY = "ridge_acc_windowed"


def load(family: str) -> list[Run]:
    """Every recorded run of one drive variant (or of the baselines)."""
    return list(iter_runs(family))


def mean(run: Run, key: str = PRIMARY) -> float:
    return run.mean(key, fallback="ridge_acc" if key == PRIMARY else None)


# The drive variant is part of every twin key. Within one drive it is constant
# and changes nothing; across drives, leaving it out would silently collapse an
# envelope run and its quadrature counterpart into one group and drop half the
# pairs. Pooling across drives is a legitimate read, so the keys have to survive it.

def twin_key(run: Run) -> tuple:
    """Everything but the geometry — identifies a run's cross-shape twins."""
    return (run.drive, run.physics, run.omega, run.cfg["damping"],
            run.cfg["clamp"], *run.condition)


def omega_twin_key(run: Run) -> tuple:
    """Everything but the natural-frequency structure."""
    return (run.drive, run.physics, run.cfg["boundary"], run.cfg["damping"],
            run.cfg["clamp"], *run.condition)


def family_twin_key(run: Run) -> tuple:
    """Everything but the physics family."""
    return (run.drive, run.cfg["boundary"], run.omega, run.cfg["damping"],
            run.cfg["clamp"], *run.condition)


def load_floors() -> dict:
    p = results_root() / "baselines" / "floors.json"
    return json.loads(p.read_text()) if p.exists() else {}


def floor_for(floors: dict, drive: str, noise_db: float, key: str = "windowed") -> float:
    entry = floors.get(f"{drive}-{noise_db:g}db")
    return 100.0 * entry[key] if entry and key in entry else math.nan


def fmt(x: float, width: int = 6, prec: int = 1, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a".rjust(width)
    return f"{x:{'+' if sign else ''}{width}.{prec}f}"


def verdict(passed: bool, bar: str) -> str:
    return f"{'MET' if passed else 'NOT MET'} (bar: {bar})"


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_baselines() -> None:
    runs = load("baselines")
    floors = load_floors()

    head("No-dynamics floors (ridge on the frontend features, field bypassed)")
    if not floors:
        print("  floors.json absent — run scripts/00_baselines.sh to compute it")
    for key in sorted(floors):
        e = floors[key]
        print(f"  {key:20s} standard {fmt(100 * e['standard'])}   "
              f"windowed {fmt(100 * e['windowed'])}")

    head("Conventional references at matched ~2k parameters (frozen-probe protocol)")
    conv = [c for c in runs if c.kind == "conventional"]
    for c in sorted(conv, key=lambda c: (c.cfg["noise_db"], c.run_id)):
        fl = floor_for(floors, "envelope", c.cfg["noise_db"])
        w = mean(c)
        print(f"  {c.run_id:44s} windowed {fmt(w)}   vs floor {fmt(w - fl, sign=True)}")

    for db in (5.0, 0.0):
        p = results_root() / "baselines" / f"conventional-trainedhead-{db:g}db.json"
        if not p.exists():
            continue
        head(f"Conventional references (trained-head protocol, {db:g} dB)")
        raw = json.loads(p.read_text())
        fl = floor_for(floors, "envelope", db)
        for arch, seeds in sorted(raw.items(), key=lambda kv: -statistics.fmean(
                r["windowed"] for r in kv[1])):
            w = 100 * statistics.fmean(r["windowed"] for r in seeds)
            best = 100 * max(r["windowed"] for r in seeds)
            over = ("n/a" if math.isnan(fl)
                    else f"{sum(1 for r in seeds if 100 * r['windowed'] > fl)}/{len(seeds)}")
            print(f"  {arch:12s} mean {fmt(w)}  best {fmt(best)}  "
                  f"vs floor {fmt(w - fl, sign=True)}   runs over floor: {over}")

    lr = [c for c in runs if c.kind == "lr-control"]
    if lr:
        head("Learning-rate control (the frozen-probe / ReLU scaffold finding)")
        for c in sorted(lr, key=lambda c: c.run_id):
            drop = statistics.fmean(r.get("loss_drop", math.nan) for r in c.rows)
            print(f"  {c.run_id:40s} windowed {fmt(mean(c))}  train-loss drop {fmt(100 * drop)}%")

    cal = [c for c in runs if c.kind == "noise-calibration"]
    if cal:
        head("Noise calibration grid (pick the level whose strongest untrained arm "
             "lands in [0.70, 0.90])")
        for c in sorted(cal, key=lambda c: (c.cfg["core"], c.cfg["noise_db"])):
            best = max((100 * r["ridge_acc"] for r in c.rows), default=math.nan)
            flag = " <- in band" if 70.0 <= best <= 90.0 else ""
            print(f"  {c.run_id:36s} strongest untrained arm {fmt(best)}{flag}")

    det = [c for c in runs if c.kind == "determinism"]
    if len(det) == 2:
        a, b = sorted(det, key=lambda c: c.run_id)
        same = a.rows and b.rows and all(
            a.rows[0].get(k) == b.rows[0].get(k) for k in (PRIMARY, "ridge_acc", "R", "plv"))
        head("Same-seed determinism")
        print(f"  two independent executions of one configuration: "
              f"{'bit-identical' if same else 'DIVERGED — investigate'}")

    ladder = results_root() / "baselines" / "readout-ladder.json"
    if ladder.exists():
        head("Readout sufficiency: is the linear probe leaving field structure behind?")
        raw = json.loads(ladder.read_text())
        print(f"  {'n':>7s}  {'field ridge':>11s} {'field mlp':>9s} {'field xf':>8s}   "
              f"{'floor ridge':>11s} {'floor mlp':>9s}   {'best edge':>9s}")
        for size in sorted(raw["run"], key=int):
            c, f = raw["run"][size], raw["floor"].get(size, {})

            def acc(entry, key):
                v = entry.get(key)
                if not isinstance(v, dict):
                    return math.nan
                return 100 * v.get("mean", v.get("test", math.nan))

            edge = max(acc(c, k) for k in ("ridge", "mlp", "transformer"))
            fedge = max(acc(f, k) for k in ("ridge", "mlp", "transformer")) if f else math.nan
            aug = " (augmented)" if c.get("augmented") else ""
            print(f"  {size:>7s}  {fmt(acc(c, 'ridge'), 11)} {fmt(acc(c, 'mlp'), 9)} "
                  f"{fmt(acc(c, 'transformer'), 8)}   {fmt(acc(f, 'ridge'), 11)} "
                  f"{fmt(acc(f, 'mlp'), 9)}   {fmt(edge - fedge, 9, sign=True)}{aug}")


def report_plateau(runs: list[Run], drive: str, floors: dict) -> None:
    head(f"Plateau: untrained {drive}-driven runs against their own floor")
    matrix = [c for c in runs if c.kind == "matrix"]
    if not matrix:
        print("  no matrix runs")
        return
    print(f"  {'condition':>14s}  {'runs':>5s}  {'mean':>6s} {'max':>6s}  "
          f"{'floor':>6s}  {'mean-floor':>10s}")
    for cond in sorted({c.condition for c in matrix}):
        sel = [c for c in matrix if c.condition == cond]
        vals = [mean(c) for c in sel]
        fl = floor_for(floors, drive, cond[0])
        label = f"{cond[0]:g} dB g{cond[1]:g}"
        print(f"  {label:>14s}  {len(sel):5d}  {fmt(statistics.fmean(vals))} "
              f"{fmt(max(vals))}  {fmt(fl)}  {fmt(statistics.fmean(vals) - fl, 10, sign=True)}")
    if drive == "carrier" and not math.isnan(floor_for(floors, drive, 0.0)):
        best = max(mean(c) for c in matrix)
        margin = best - floor_for(floors, drive, 0.0)
        print(f"\n  transduction: best run {fmt(best)} = floor {fmt(margin, sign=True)}  "
              f"{verdict(margin >= BAR_TRANSDUCTION, f'+{BAR_TRANSDUCTION:g} to claim restoration')}")


def twin_deltas(matrix: list[Run], key_fn, split: str,
                reference: str) -> dict[str, list[float]]:
    """Per-level deltas against a reference level, over runs identical in all else.

    `key_fn` collapses everything EXCEPT the factor under test, so two runs that
    share a key differ in exactly one thing. That is the whole discipline: a
    geometry delta is only a geometry delta if its two runs are twins.
    """
    groups: dict[tuple, dict[str, Run]] = {}
    for c in matrix:
        groups.setdefault(key_fn(c), {})[
            getattr(c, split) if split != "boundary" else c.cfg["boundary"]] = c
    out: dict[str, list[float]] = {}
    for level_map in groups.values():
        ref = level_map.get(reference)
        if ref is None:
            continue
        for level, run in level_map.items():
            if level != reference:
                out.setdefault(level, []).append(mean(run) - mean(ref))
    return out


def per_condition_means(matrix: list[Run], key_fn, split: str,
                        reference: str) -> dict[tuple, dict[str, float]]:
    """The same twin deltas, read one operating condition at a time.

    Pooling hides sign flips: a factor can average to nothing overall while
    running one way at one condition and the other way at another. The paper
    quotes the per-condition extremes, so the scorer has to print them.
    """
    out = {}
    for cond in sorted({c.condition for c in matrix}):
        deltas = twin_deltas([c for c in matrix if c.condition == cond],
                             key_fn, split, reference)
        out[cond] = {k: statistics.fmean(v) for k, v in deltas.items()}
    return out


def report_geometry(runs: list[Run]) -> None:
    matrix = [c for c in runs if c.kind == "matrix"]
    shapes = {c.cfg["boundary"] for c in matrix}
    if len(shapes) < 2:
        return
    head("Geometry: every shape against its torus twin")
    deltas = twin_deltas(matrix, twin_key, "boundary", "torus")
    worst = 0.0
    for shape in GEOMETRIES:
        d = deltas.get(shape)
        if not d:
            continue
        m = statistics.fmean(d)
        worst = max(worst, abs(m))
        pos = 100 * sum(1 for x in d if x > 0) / len(d)
        print(f"  {shape:10s} n={len(d):4d}  mean {fmt(m, sign=True)}  "
              f"range [{fmt(min(d), sign=True)}, {fmt(max(d), sign=True)}]  {pos:3.0f}% positive")
    by_cond = per_condition_means(matrix, twin_key, "boundary", "torus")
    if len(by_cond) > 1:
        print(f"\n  per condition (pooling hides sign flips; n = {len(deltas[next(iter(deltas))]) // len(by_cond)}"
              f" twins per shape per condition):")
        shown = [s for s in GEOMETRIES if s in deltas]
        print("    " + " " * 12 + "".join(f"{s[:7]:>9s}" for s in shown))
        for cond, means in by_cond.items():
            print(f"    {cond[0]:>4g} dB g{cond[1]:<4g}"
                  + "".join(f"{means.get(s, math.nan):+9.2f}" for s in shown))
        flat = [(m, s, c) for c, ms in by_cond.items() for s, m in ms.items()]
        lo, hi = min(flat), max(flat)
        print(f"    extremes: {lo[1]} {lo[0]:+.2f} at {lo[2][0]:g} dB g{lo[2][1]:g}, "
              f"{hi[1]} {hi[0]:+.2f} at {hi[2][0]:g} dB g{hi[2][1]:g}")
    print(f"\n  {verdict(worst >= BAR_GEOMETRY, f'a shape must beat torus by +{BAR_GEOMETRY:g}')}")


def report_omega(runs: list[Run]) -> None:
    matrix = [c for c in runs if c.kind == "matrix"]
    if not matrix:
        return
    head("Tonotopic design: designed omega against its randomized twin")
    deltas = twin_deltas(matrix, omega_twin_key, "omega", "random")
    best = -math.inf
    for level in ("designed", "uniform"):
        d = deltas.get(level)
        if not d:
            continue
        m = statistics.fmean(d)
        if level == "designed":
            best = m
        print(f"  {level:10s} n={len(d):4d}  mean {fmt(m, sign=True)}  "
              f"range [{fmt(min(d), sign=True)}, {fmt(max(d), sign=True)}]")
    print("\n  designed - random per physics family (pooled over conditions):")
    for fam in FAMILIES:
        sub = [c for c in matrix if c.physics == fam]
        d = twin_deltas(sub, omega_twin_key, "omega", "random").get("designed")
        if d:
            per_cond = per_condition_means(sub, omega_twin_key, "omega", "random")
            signs = "".join("+" if cm.get("designed", 0) > 0 else "-"
                            for cm in per_cond.values())
            print(f"    {fam:12s} n={len(d):3d}  mean {fmt(statistics.fmean(d), sign=True)}"
                  f"   per-condition signs {signs}")
    by_cond = per_condition_means(matrix, omega_twin_key, "omega", "random")
    print("  designed - random per condition (all families):")
    for cond, means in by_cond.items():
        print(f"    {cond[0]:>4g} dB g{cond[1]:<4g} {fmt(means.get('designed', math.nan), sign=True)}")
    print(f"\n  {verdict(best >= BAR_DESIGNED_OMEGA, f'designed must beat random by +{BAR_DESIGNED_OMEGA:g}')}")


def report_family(runs: list[Run]) -> None:
    matrix = [c for c in runs if c.kind == "matrix"]
    if not matrix:
        return
    head("Coupling law: every physics family against its kuramoto twin")
    deltas = twin_deltas(matrix, family_twin_key, "physics", "kuramoto")
    spread = 0.0
    for fam in FAMILIES:
        d = deltas.get(fam)
        if not d:
            continue
        m = statistics.fmean(d)
        spread = max(spread, abs(m))
        print(f"  {fam:12s} n={len(d):4d}  mean {fmt(m, sign=True)}  "
              f"range [{fmt(min(d), sign=True)}, {fmt(max(d), sign=True)}]")
    print(f"\n  {verdict(spread >= BAR_FAMILY_SPREAD, f'a family must move +/-{BAR_FAMILY_SPREAD:g}')}")


def report_settle(runs: list[Run]) -> None:
    matrix = [c for c in runs if c.kind == "matrix" and
              any("ridge_acc_settle" in r for r in c.rows)]
    if not matrix:
        return
    head("Settle versus stream: drive-free ring-down against the driven trajectory")
    d = [mean(c, "ridge_acc_settle") - mean(c) for c in matrix]
    pos = 100 * sum(1 for x in d if x > 0) / len(d)
    print(f"  n={len(d)}  mean {fmt(statistics.fmean(d), sign=True)}  {pos:.0f}% positive "
          f"-> the information is in the DRIVEN evolution")


def report_gain(runs: list[Run]) -> None:
    gates = [c for c in runs if c.kind == "gain"]
    if not gates:
        return
    head("Gain response (gate size; effective drive = gain x row RMS)")
    for c in sorted(gates, key=lambda c: (c.physics, c.cfg["gain"], c.omega, c.run_id)):
        seeds = len(c.rows)
        kick = c.instrument("kick_max")
        note = "" if math.isnan(kick) else f"  max drive increment {kick:.2f} rad"
        print(f"  {c.run_id:44s} windowed {fmt(mean(c))}  ({seeds} seed"
              f"{'s' if seeds > 1 else ''}){note}")

    matrix = [c for c in runs if c.kind == "matrix"]
    conds = sorted({c.condition for c in matrix})
    if len(conds) > 1:
        base = [c for c in matrix if c.condition == conds[-1]]
        for cond in conds[:-1]:
            other = {(c.physics, c.cfg["boundary"], c.omega, c.cfg["damping"], c.cfg["clamp"]): c
                     for c in matrix if c.condition == cond}
            d = [mean(other[k]) - mean(c)
                 for c in base
                 if (k := (c.physics, c.cfg["boundary"], c.omega,
                           c.cfg["damping"], c.cfg["clamp"])) in other]
            if d:
                pos = 100 * sum(1 for x in d if x > 0) / len(d)
                print(f"\n  matrix delta {cond[0]:g}dB/g{cond[1]:g} - "
                      f"{conds[-1][0]:g}dB/g{conds[-1][1]:g}: n={len(d)} "
                      f"mean {fmt(statistics.fmean(d), sign=True)}  {pos:.0f}% positive")


def report_coherence(runs: list[Run], drive: str, make_figure: bool) -> None:
    matrix = [c for c in runs if c.kind == "matrix" and not math.isnan(c.instrument("R"))]
    if len(matrix) < 8:
        return
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("\n  (scipy absent — skipping the coherence correlation)")
        return
    head("Coherence and readability: does readability anti-correlate with order?")
    hits, series = 0, {}
    for cond in sorted({c.condition for c in matrix}):
        sel = [c for c in matrix if c.condition == cond]
        rho = spearmanr([c.instrument("R") for c in sel], [mean(c) for c in sel]).statistic
        hits += rho <= BAR_COHERENCE_RHO
        series[cond] = sel
        print(f"  {cond[0]:g} dB g{cond[1]:g}  n={len(sel):4d}  rho {fmt(rho, 6, 3, sign=True)}")
    print(f"\n  {verdict(hits >= 2, f'rho <= {BAR_COHERENCE_RHO} in at least 2 of 3 conditions')}")
    if make_figure and drive == "envelope":
        write_coherence_figure(series)


def write_coherence_figure(series: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib absent — skipping the figure)")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ("#534AB7", "#0F6E56", "#D85A30")
    for cond, color in zip(sorted(series), colors, strict=False):
        runs = series[cond]
        ax.scatter([c.instrument("R") for c in runs], [mean(c) / 100 for c in runs],
                   s=8, alpha=0.45, color=color, linewidths=0,
                   label=f"{cond[0]:g} dB, gain {cond[1]:g} (n={len(runs)})")
    ax.set_xlabel("order parameter R (global coherence)")
    ax.set_ylabel("windowed accuracy")
    ax.set_title("Untrained runs: readability vs coherence")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    # Both, and for the same reason the authored figures ship both: the LaTeX
    # builds take the vector PDF, because a 200 dpi scatter in a submission PDF
    # pixelates the moment a reviewer zooms, and everything else takes the PNG.
    stem = FIGURES_DIR / "g9-readability-vs-coherence"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 200})):
        fig.savefig(stem.with_suffix(suffix), **kwargs)
    print(f"  wrote {stem}.{{pdf,png}}")


def report_sweeps(runs: list[Run]) -> None:
    for kind, title in (("coupling", "Coupling structure: circulant against random graphs"),
                        ("clamp", "Spectral clamp x pinning grid"),
                        ("sensitivity", "Sensitivity of the canonical physics values")):
        sel = [c for c in runs if c.kind == kind]
        if not sel:
            continue
        head(title)
        for c in sorted(sel, key=lambda c: c.run_id):
            spread = ""
            if len(c.rows) > 1:
                vals = [100 * r[PRIMARY] for r in c.rows]
                spread = f"  +/-{(max(vals) - min(vals)) / 2:.1f}"
            print(f"  {c.run_id:44s} windowed {fmt(mean(c))}{spread}  "
                  f"R {c.instrument('R'):.3f}")


def report_order(runs: list[Run]) -> None:
    """The order task is scored on the FULL-SPAN pooled ridge, never the
    windowed one: the whole construction rests on the read being order-free,
    and per-window statistics are not. Any accuracy above chance here must come
    from state that persists across time."""
    order = [c for c in runs if c.kind == "order"]
    if not order:
        return
    head("Order discrimination — full-span pooled read (chance 0.500, "
         "order-free by construction)")
    pairs = sorted({c.run_id.split("-")[0] for c in order})
    arms = sorted({c.run_id.split("-", 1)[1] for c in order})
    print(f"  {'pair':>6s}  " + "  ".join(f"{a:>13s}" for a in arms))
    passing, coupling_gain = 0, []
    for pair in pairs:
        row, best, per_arm = [], 0.0, {}
        for arm in arms:
            c = next((c for c in order if c.run_id == f"{pair}-{arm}"), None)
            if c is None:
                row.append("n/a".rjust(13))
                continue
            vals = [100 * r["ridge_acc"] for r in c.rows]
            m = statistics.fmean(vals)
            per_arm[arm] = m
            if arm not in ("gru", "severed"):
                best = max(best, m)
            row.append(f"{m:8.1f}+/-{(max(vals) - min(vals)) / 2:3.1f}")
        if "severed" in per_arm and "kuramoto" in per_arm:
            coupling_gain.append(per_arm["kuramoto"] - per_arm["severed"])
        passing += best >= 100 * BAR_ORDER_ACC
        print(f"  {pair.replace('pair', ''):>6s}  " + "  ".join(row))
    print(f"\n  {verdict(passing >= 3, f'>= {BAR_ORDER_ACC} on at least 3 of 5 pairs')} "
          f"({passing}/{len(pairs)} pairs)")
    if coupling_gain:
        print(f"  severed-coupling control: the network adds "
              f"{min(coupling_gain):+.1f} to {max(coupling_gain):+.1f} over uncoupled "
              f"oscillators -> the memory is predominantly per-oscillator integration")


def report_drive(drive: str, make_figure: bool) -> None:
    runs = load(drive)
    if not runs:
        print(f"no runs under results/{drive}")
        return
    floors = load_floors()
    print(f"\n{'=' * 72}\n{drive.upper()} DRIVE — {len(runs)} runs\n{'=' * 72}")
    report_plateau(runs, drive, floors)
    report_geometry(runs)
    report_omega(runs)
    report_family(runs)
    report_settle(runs)
    report_gain(runs)
    report_coherence(runs, drive, make_figure)
    report_sweeps(runs)
    report_order(runs)


def report_across_drives() -> None:
    """The two reads the paper pools across drives at a fixed condition."""
    matrix = [c for drive in DRIVES for c in load(drive) if c.kind == "matrix"]
    if not matrix:
        return
    print(f"\n{'=' * 72}\nACROSS DRIVES (envelope + quadrature + carrier)\n{'=' * 72}")
    head("Settle versus stream, per condition")
    for cond in sorted({c.condition for c in matrix}):
        sel = [c for c in matrix
               if c.condition == cond and any("ridge_acc_settle" in r for r in c.rows)]
        if not sel:
            continue
        d = [mean(c, "ridge_acc_settle") - mean(c) for c in sel]
        pos = 100 * sum(1 for x in d if x > 0) / len(d)
        print(f"  {cond[0]:g} dB g{cond[1]:g}  n={len(d):4d}  "
              f"mean {fmt(statistics.fmean(d), sign=True)}  {pos:.0f}% positive")

    head("Phase-referenced (quadrature) drive against its magnitude twin")
    env = {(c.physics, c.cfg["boundary"], c.omega, c.cfg["damping"], c.cfg["clamp"],
            *c.condition): c for c in matrix if c.family == "envelope"}
    d = [mean(c) - env[k].mean() for c in matrix if c.family == "quadrature"
         and (k := (c.physics, c.cfg["boundary"], c.omega, c.cfg["damping"],
                    c.cfg["clamp"], *c.condition)) in env]
    if d:
        neg = 100 * sum(1 for x in d if x < 0) / len(d)
        print(f"  n={len(d)}  mean {fmt(statistics.fmean(d), sign=True)}  "
              f"{neg:.0f}% below their magnitude twin")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="score the results tree")
    ap.add_argument("target", nargs="?", default="all",
                    choices=("all", "baselines", *DRIVES))
    ap.add_argument("--figures", action="store_true",
                    help="also regenerate the figures derived from these results")
    a = ap.parse_args(argv)
    if a.target in ("all", "baselines"):
        print(f"{'=' * 72}\nBASELINES\n{'=' * 72}")
        report_baselines()
    for drive in DRIVES:
        if a.target in ("all", drive):
            report_drive(drive, a.figures)
    if a.target == "all":
        report_across_drives()


if __name__ == "__main__":
    main()

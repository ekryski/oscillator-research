"""Contracts for `harness.measurement.score`.

Scoring is where a bar becomes a verdict, so the arithmetic under it has to be
pinned: a twin comparison must differ in exactly one factor, the order task
must be read through the order-free column and never the windowed one, and a
missing floor must surface as "unavailable" rather than as a silent zero.

These run against the committed results, so they also act as a tripwire on the
record itself: if a run's stored key stops matching the configuration it
recorded, the twin groupings collapse and these fail.
"""

import math
import statistics
import sys

import pytest

sys.path.insert(0, ".")
from harness.measurement import score
from harness.results import Run


def run(**cfg) -> Run:
    base = dict(core="phase", coupling="kuramoto", boundary="torus", arms="frozen",
                omega_uniform=False, damping=0.3, clamp=1.0, noise_db=0.0, gain=2.0)
    base.update(cfg)
    return Run("envelope/matrix-kuramoto.json", "x", base,
               [{"ridge_acc_windowed": 0.8, "ridge_acc": 0.7}])


def test_physics_reads_the_core_before_the_coupling_law():
    assert run().physics == "kuramoto"
    assert run(coupling="winfree").physics == "winfree"
    assert run(core="sl").physics == "sl"
    assert run(core="sl-fixedamp").physics == "sl-fixedamp"
    # the SL cores are built on the kuramoto coupling term; the core wins
    assert run(core="sl", coupling="kuramoto").physics == "sl"
    assert run(core="randgraph", graph_k=10).physics == "randgraph-k10"


def test_omega_level_distinguishes_all_three_structures():
    assert run().omega == "random"
    assert run(arms="designed").omega == "designed"
    assert run(omega_uniform=True).omega == "uniform"
    assert run(arms="designed", omega_uniform=True).omega == "uniform"


def test_twin_keys_hold_everything_but_the_factor_under_test():
    a, b = run(boundary="torus"), run(boundary="sphere")
    assert score.twin_key(a) == score.twin_key(b), "geometry twins differ only in shape"
    assert score.omega_twin_key(a) != score.omega_twin_key(b), "but they are not omega twins"
    assert score.omega_twin_key(run()) == score.omega_twin_key(run(arms="designed"))
    assert score.family_twin_key(run()) == score.family_twin_key(run(coupling="winfree"))
    assert score.twin_key(run()) != score.twin_key(run(gain=1.0)), "conditions never mix"


def test_mean_falls_back_to_the_standard_column_when_windowed_is_absent():
    r = Run("baselines/lr-control.json", "cnn-lr0.003-5db",
            {"core": "phase", "coupling": "kuramoto", "boundary": "torus", "arms": "cnn",
             "damping": 0.5, "clamp": 0.5, "noise_db": 5.0, "gain": 2.0},
            [{"ridge_acc": 0.5}])
    assert score.mean(r) == pytest.approx(50.0)
    assert math.isnan(r.mean("ridge_acc_settle"))


def test_missing_floor_is_reported_as_unavailable_not_as_zero():
    assert math.isnan(score.floor_for({}, "envelope", 0.0))
    assert score.fmt(score.floor_for({}, "envelope", 0.0)).strip() == "n/a"
    floors = {"envelope-0db": {"standard": 0.7, "windowed": 0.805}}
    assert score.floor_for(floors, "envelope", 0.0) == pytest.approx(80.5)
    # a run is only ever comparable to ITS OWN representation's floor
    assert math.isnan(score.floor_for(floors, "carrier", 0.0))


def test_verdict_names_the_bar_it_was_scored_against():
    assert score.verdict(True, "+3").startswith("MET")
    assert "bar: +3" in score.verdict(False, "+3")
    assert score.verdict(False, "+3").startswith("NOT MET")


# --- against the committed record -------------------------------------------

def test_the_committed_record_loads_and_is_grouped_by_drive():
    counts = {d: len(score.load(d)) for d in score.DRIVES}
    assert all(counts.values()), counts
    assert counts["envelope"] > counts["carrier"], "the carrier arm is a diagonal"
    assert score.load("baselines"), "baselines are the reference frame"


def test_matrix_runs_are_complete_twin_families():
    """Every non-torus matrix run has a torus twin, or a geometry delta would
    be computed against a hole."""
    matrix = [r for r in score.load("envelope") if r.kind == "matrix"]
    torus = {score.twin_key(r) for r in matrix if r.cfg["boundary"] == "torus"}
    orphans = [r.run_id for r in matrix
               if r.cfg["boundary"] != "torus" and score.twin_key(r) not in torus]
    assert not orphans, orphans[:5]


def test_geometry_deltas_are_computed_over_one_factor_only():
    matrix = [r for r in score.load("envelope") if r.kind == "matrix"]
    deltas = score.twin_deltas(matrix, score.twin_key, "boundary", "torus")
    assert set(deltas) == set(score.GEOMETRIES) - {"torus"}
    assert all(len(v) > 100 for v in deltas.values()), {k: len(v) for k, v in deltas.items()}


def test_order_runs_are_the_order_task_and_binary():
    """The task is built so an order-free readout provably cannot answer it, so
    every order run has to actually be that task at two classes."""
    order = [r for r in score.load("envelope") if r.kind == "order"]
    assert len(order) >= 15
    for r in order:
        assert r.cfg["task"] == "digitpairs"
        assert r.cfg["n_classes"] == 2


def test_twin_keys_never_pair_runs_from_different_drives():
    """Pooling across drives is a legitimate read (the paper's settle and
    quadrature-collapse numbers are exactly that), and a key that omitted the
    drive would collapse an envelope run into its quadrature counterpart and
    silently halve the pairs."""
    env = [r for r in score.load("envelope") if r.kind == "matrix"]
    quad = [r for r in score.load("quadrature") if r.kind == "matrix"]
    pooled = score.twin_deltas(env + quad, score.twin_key, "boundary", "torus")
    per_drive = sum(len(v) for v in
                    score.twin_deltas(env, score.twin_key, "boundary", "torus").values())
    per_drive += sum(len(v) for v in
                     score.twin_deltas(quad, score.twin_key, "boundary", "torus").values())
    assert sum(len(v) for v in pooled.values()) == per_drive


# --- the counts the paper states --------------------------------------------

def test_the_factorial_is_the_size_the_abstract_claims():
    """The abstract quotes a run count for the two-pathway factorial. It is a
    claim about the record like any other, so it gets a test."""
    env = [r for r in score.load("envelope") if r.kind == "matrix"]
    quad = [r for r in score.load("quadrature") if r.kind == "matrix"]
    assert len(env) + len(quad) == 1800
    for pool in (env, quad):
        per_condition = {c: sum(1 for r in pool if r.condition == c)
                         for c in {r.condition for r in pool}}
        assert len(set(per_condition.values())) == 1, "the factorial must be balanced"
    # section 5.2's "600 supported runs each"
    both = env + quad
    for condition in {r.condition for r in both}:
        assert sum(1 for r in both if r.condition == condition) == 600


def test_the_coherence_scatter_is_the_size_section_5_6_claims():
    env = [r for r in score.load("envelope") if r.kind == "matrix"]
    assert len(env) == 936
    assert all(sum(1 for r in env if r.condition == c) == 312
               for c in {r.condition for r in env})


def test_the_carrier_arm_is_the_fourteen_run_diagonal():
    carrier = [r for r in score.load("carrier") if r.kind == "matrix"]
    assert len(carrier) == 14


def test_the_whole_record_is_the_size_section_9_claims():
    assert sum(len(score.load(f)) for f in (*score.DRIVES, "baselines")) == 1940


# --- the effect sizes section 5.2 states -----------------------------------
#
# These are the paper's scored claims, not bookkeeping: each asserts both the
# value AND the scope it was read at, because an effect size without its
# pooling is not a number anyone can check.

def envelope_matrix():
    return [r for r in score.load("envelope") if r.kind == "matrix"]


def test_geometry_twins_are_48_per_shape_per_condition_and_144_pooled():
    matrix = envelope_matrix()
    pooled = score.twin_deltas(matrix, score.twin_key, "boundary", "torus")
    assert set(pooled) == set(score.GEOMETRIES) - {"torus"}
    assert {len(v) for v in pooled.values()} == {144}, "coverage must be uniform per shape"
    per_cond = score.per_condition_means(matrix, score.twin_key, "boundary", "torus")
    assert len(per_cond) == 3
    for cond in per_cond:
        one = score.twin_deltas([r for r in matrix if r.condition == cond],
                                score.twin_key, "boundary", "torus")
        assert {len(v) for v in one.values()} == {48}


def test_geometry_effect_sizes_match_the_paper():
    matrix = envelope_matrix()
    pooled = {k: statistics.fmean(v) for k, v in
              score.twin_deltas(matrix, score.twin_key, "boundary", "torus").items()}
    # "the per-shape mean deltas run from -0.13 (sphere) to +0.39 (helix)"
    assert min(pooled, key=pooled.get) == "sphere"
    assert max(pooled, key=pooled.get) == "helix"
    assert pooled["sphere"] == pytest.approx(-0.13, abs=0.01)
    assert pooled["helix"] == pytest.approx(+0.39, abs=0.01)
    # "helix ... is positive in 61% of its 144 twins"
    helix = score.twin_deltas(matrix, score.twin_key, "boundary", "torus")["helix"]
    assert round(100 * sum(1 for x in helix if x > 0) / len(helix)) == 61
    # "the extremes widen only to -0.63 (sphere, +5 dB g=1) and +0.58 (helix, +5 dB g=2)"
    flat = [(m, s, c) for c, ms in
            score.per_condition_means(matrix, score.twin_key, "boundary", "torus").items()
            for s, m in ms.items()]
    lo, hi = min(flat), max(flat)
    assert (round(lo[0], 2), lo[1], lo[2]) == (-0.63, "sphere", (5.0, 1.0))
    assert (round(hi[0], 2), hi[1], hi[2]) == (+0.58, "helix", (5.0, 2.0))
    # no shape anywhere near the bar
    assert max(abs(m) for m, _, _ in flat) < score.BAR_GEOMETRY


def test_designed_omega_effect_sizes_match_the_paper():
    matrix = envelope_matrix()
    # "from +0.28 (kuramoto) down to -1.21 (harmonic2)", 72 twins per phase family
    per_family = {}
    for fam in score.FAMILIES:
        sub = [r for r in matrix if r.physics == fam]
        d = score.twin_deltas(sub, score.omega_twin_key, "omega", "random")["designed"]
        per_family[fam] = (len(d), statistics.fmean(d))
    assert per_family["kuramoto"][0] == 72 and per_family["sl"][0] == 12
    assert per_family["kuramoto"][1] == pytest.approx(+0.28, abs=0.01)
    assert per_family["harmonic2"][1] == pytest.approx(-1.21, abs=0.01)
    assert max(m for _, m in per_family.values()) == pytest.approx(+0.28, abs=0.01)
    assert min(m for _, m in per_family.values()) == pytest.approx(-1.21, abs=0.01)
    # "harmonic2 and winfree negative in all three conditions"
    for fam in ("harmonic2", "winfree"):
        sub = [r for r in matrix if r.physics == fam]
        signs = score.per_condition_means(sub, score.omega_twin_key, "omega", "random")
        assert all(m["designed"] < 0 for m in signs.values()), fam
    # "pooling families within a condition (104 twins), -0.31, -0.42 and -0.56"
    by_cond = score.per_condition_means(matrix, score.omega_twin_key, "omega", "random")
    got = sorted(round(m["designed"], 2) for m in by_cond.values())
    assert got == [-0.56, -0.42, -0.31]
    for cond in by_cond:
        d = score.twin_deltas([r for r in matrix if r.condition == cond],
                              score.omega_twin_key, "omega", "random")["designed"]
        assert len(d) == 104


def test_the_widest_omega_read_is_200_twins_and_matches_the_paper():
    """'all families and both drive pathways at one condition (200 twins)' —
    the widest read in the paper, and the one the old text mislabelled."""
    both = [r for r in score.load("envelope") + score.load("quadrature")
            if r.kind == "matrix"]
    by_cond = score.per_condition_means(both, score.omega_twin_key, "omega", "random")
    for cond in by_cond:
        d = score.twin_deltas([r for r in both if r.condition == cond],
                              score.omega_twin_key, "omega", "random")["designed"]
        assert len(d) == 200
    assert sorted(round(m["designed"], 2) for m in by_cond.values()) == [-0.60, -0.12, 0.52]


def test_the_carrier_diagonal_omega_and_geometry_deltas_match_the_paper():
    car = [r for r in score.load("carrier") if r.kind == "matrix"]
    om = score.twin_deltas(car, score.omega_twin_key, "omega", "random")["designed"]
    assert len(om) == 7
    assert statistics.fmean(om) == pytest.approx(-2.0, abs=0.05)
    # "the same to one decimal over the six torus families alone"
    torus_only = [r for r in car if r.cfg["boundary"] == "torus"]
    fams = [statistics.fmean(score.twin_deltas(
        [r for r in torus_only if r.physics == f], score.omega_twin_key,
        "omega", "random")["designed"]) for f in {r.physics for r in torus_only}]
    assert len(fams) == 6
    assert round(statistics.fmean(fams), 1) == round(statistics.fmean(om), 1) == -2.0
    helix = score.twin_deltas(car, score.twin_key, "boundary", "torus")["helix"]
    assert sorted(round(x, 1) for x in helix) == [-1.0, 1.0]

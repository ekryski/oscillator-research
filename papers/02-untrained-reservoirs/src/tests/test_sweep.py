"""Contracts for `harness.sweep` — the experiment plan.

The grids in this module ARE the description of what was run, so the test that
matters is that they reproduce the committed record exactly: every planned run
already recorded, and every recorded run still planned. If those two diverge,
either the code no longer describes the experiment or the record has a gap, and
both are the kind of thing a reviewer is entitled to catch.
"""

import sys

import pytest

sys.path.insert(0, ".")
from harness import results as store
from harness import sweep


def planned(name: str) -> list:
    return list(sweep.SWEEPS[name]())


def addresses(name: str) -> set[tuple[str, str]]:
    return {store.address(sweep._config_of(r), r.kind) for r in planned(name)}


@pytest.mark.parametrize("name", list(sweep.SWEEPS))
def test_the_plan_reproduces_the_committed_record(name):
    want = addresses(name)
    have = {(r.group, r.run_id) for r in store.iter_runs(name)}
    assert not want - have, f"planned but never recorded: {sorted(want - have)[:5]}"
    assert not have - want, f"recorded but no longer planned: {sorted(have - want)[:5]}"


@pytest.mark.parametrize("name", list(sweep.SWEEPS))
def test_no_planned_run_collides_with_another(name):
    assert len(addresses(name)) == len(planned(name)), "two planned runs share an address"


def test_the_factorial_is_the_size_the_paper_states():
    """6 physics x 6 geometries x 3 frequency structures x 2 pinning x 2 clamp,
    at 3 operating points — minus the documented amplitude-core torus scope."""
    matrix = [r for r in planned("envelope") if r.kind == "matrix"]
    per_condition = len(matrix) // len(sweep.CONDITIONS)
    phase = len(sweep.PHASE_FAMILIES) * len(sweep.SHAPES) * len(sweep.OMEGAS) * 2 * 2
    sl = len(sweep.SL_FAMILIES) * 1 * len(sweep.OMEGAS) * 2 * 2
    assert per_condition == phase + sl == 312
    # the quadrature pathway has no Adler hook for the amplitude cores
    quad = [r for r in planned("quadrature") if r.kind == "matrix"]
    assert len(quad) // len(sweep.CONDITIONS) == phase == 288


def test_canonical_physics_values_are_nonzero_where_they_have_to_be():
    """Zero alpha and beta are exactly kuramoto, so an untrained sakaguchi or
    harmonic2 run at zero would not be a distinct factor level at all."""
    assert sweep.ALPHA > 0 and sweep.BETA > 0
    sak = sweep.physics_flags("sakaguchi")
    assert float(sak[sak.index("--sakaguchi-alpha") + 1]) > 0
    har = sweep.physics_flags("harmonic2")
    assert float(har[har.index("--harmonic2-beta") + 1]) > 0


def test_resource_planning_respects_cores_and_memory():
    workers, threads = sweep.plan_resources("envelope", None, None)
    assert workers >= 1 and threads >= 1
    # the carrier pathway is sample-rate: its activations are far larger, so it
    # must never plan more workers than the envelope pathway on the same box
    assert sweep.plan_resources("carrier", None, None)[0] <= workers
    assert sweep.plan_resources("envelope", 3, 2) == (3, 2), "explicit values win"


def test_memory_probes_return_something_usable():
    assert sweep.total_memory_gb() > 0.5
    assert sweep.available_memory_gb() > 0.0


def test_progress_reports_counts_and_failures():
    p = sweep.Progress(total=4, tty=False)
    for ok in (True, True, False, True):
        p.update("a-run", ok)
    assert p.done == 4 and p.failed == 1


def test_human_durations_stay_readable():
    assert sweep.human(45) == "45s"
    assert sweep.human(600) == "10m"
    assert sweep.human(7200) == "2.0h"

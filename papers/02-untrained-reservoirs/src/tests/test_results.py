"""Contracts for `harness.results` — the results record.

Two properties carry the whole record. A run's address must be a pure function
of the configuration it recorded, so the same experiment always lands in the
same place no matter which process or shard produced it. And writes must
survive concurrency, because a sweep runs many workers that finish into the
same group file at once.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

sys.path.insert(0, ".")
from harness import results as store


def cfg(**over) -> dict:
    base = dict(frontend="mag", core="phase", coupling="kuramoto", boundary="torus",
                arms="frozen", omega_uniform=False, damping=0.3, clamp=1.0,
                noise_db=0.0, gain=2.0, seeds="0", lr=0.003)
    base.update(over)
    return base


def test_address_is_a_pure_function_of_the_configuration():
    a = store.address(cfg())
    assert a == store.address(cfg()), "the same configuration must address the same place"
    assert a == ("envelope/matrix-kuramoto.json", "torus-random-lam0.3-clamp1-0db-g2")


def test_the_group_file_is_the_coupling_law_within_the_drive():
    assert store.address(cfg())[0] == "envelope/matrix-kuramoto.json"
    assert store.address(cfg(coupling="winfree"))[0] == "envelope/matrix-winfree.json"
    assert store.address(cfg(core="sl"))[0] == "envelope/matrix-sl.json"
    assert store.address(cfg(frontend="quad"))[0] == "quadrature/matrix-kuramoto.json"
    assert store.address(cfg(frontend="carrier"))[0] == "carrier/matrix-kuramoto.json"


def test_every_factor_of_the_matrix_appears_in_the_run_id():
    """A run id that dropped a factor would silently collide two experiments."""
    base = store.address(cfg())[1]
    for change in (dict(boundary="sphere"), dict(arms="designed"), dict(omega_uniform=True),
                   dict(damping=0.1), dict(clamp=0.5), dict(noise_db=5.0), dict(gain=1.0)):
        assert store.address(cfg(**change))[1] != base, change


def test_the_kind_separates_runs_the_physics_cannot():
    """A gain gate and a matrix point can share every physics value; only what
    the run is FOR tells them apart, which is why the kind is recorded."""
    shared = cfg(gain=8.0)
    assert store.address(shared, "matrix")[0] != store.address(shared, "gain")[0]
    with pytest.raises(ValueError, match="unknown run kind"):
        store.address(shared, "nonsense")


def test_severed_coupling_is_addressed_as_zero_not_as_scientific_notation():
    _, run_id = store.address(cfg(clamp=store.SEVERED_CLAMP, task="digitpairs",
                                  pair="3,7"), "order")
    assert run_id == "pair37-severed"


def test_write_then_read_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    store.write_run(cfg(), [{"ridge_acc": 0.5}])
    group, run_id = store.address(cfg())
    data = json.loads((tmp_path / group).read_text())
    assert list(data["runs"]) == [run_id]
    assert data["runs"][run_id]["rows"] == [{"ridge_acc": 0.5}]
    assert data["runs"][run_id]["config"]["kind"] == "matrix"


def test_has_run_is_the_resume_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    assert not store.has_run(cfg())
    store.write_run(cfg(), [{"ridge_acc": 0.5}])
    assert store.has_run(cfg())
    assert not store.has_run(cfg(boundary="sphere"))


def test_concurrent_writers_do_not_lose_runs(tmp_path, monkeypatch):
    """The real failure mode this guards: many workers finishing at once into
    one group file, each overwriting the others' merge."""
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    shapes = ["torus", "cylinder", "sheet", "helix", "cube", "sphere"] * 4
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: store.write_run(
            cfg(boundary=shapes[i], damping=0.1 + 0.01 * i), [{"ridge_acc": 0.1 * i}]),
            range(len(shapes))))
    data = json.loads((tmp_path / "envelope" / "matrix-kuramoto.json").read_text())
    assert len(data["runs"]) == len(shapes), "a concurrent write was lost"


def test_the_committed_record_addresses_itself_consistently():
    """The stored key and the key derived from the stored configuration can
    never drift apart — this is what keeps the record self-describing."""
    seen = 0
    for run in store.iter_runs():
        group, run_id = store.address(run.cfg, run.cfg.get("kind"))
        assert (group, run_id) == (run.group, run.run_id), run.name
        seen += 1
    assert seen > 1900, f"expected the full record, saw {seen} runs"


def test_the_committed_record_has_no_duplicate_runs():
    names = [r.name for r in store.iter_runs()]
    assert len(names) == len(set(names))

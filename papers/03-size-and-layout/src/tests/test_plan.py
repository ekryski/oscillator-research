"""Paper 03's tiers: what they run, what they take from paper 02, and what they cost."""

import pytest

from harness.confirm import plan
from harness.confirm import run as rn
from harness.confirm.arms import Arm

#: the run counts TIERS.md and REGISTRATION.md state
COUNTS = {"gate": 30, "size": 1674, "trained": 1350, "design": 13500, "quadrature": 1134, "carrier": 240,
          "design-quadrature": 12420, "design-carrier": 1875}
REUSED = {"gate": 0, "size": 54, "trained": 30, "design": 300, "quadrature": 18, "carrier": 9,
          "design-quadrature": 48, "design-carrier": 18}


@pytest.mark.parametrize("tier", plan.TIERS)
def test_each_tier_has_the_documented_runs_and_no_two_share_an_address(tier):
    specs = plan.planned([tier])
    assert len(specs) == COUNTS[tier]
    assert sum(map(plan.reused, specs)) == REUSED[tier]


def test_the_lattices_are_nine_and_16x16_runs_once():
    assert list(plan.lattices()) == [(8, 0), (8, 16), (16, 0), (32, 0), (32, 16), (64, 0), (64, 16), (128, 0),
                                     (128, 16)]


def test_the_reused_cells_are_paper_02s_16x16_4_channel_runs_under_their_own_ids():
    s = rn.Spec("size", "recognition", "envelope", 0.0, 1.0, 2, Arm("field"))
    assert plan.paper02_group(s) == "tier1-recognition-envelope"
    assert s.run_id() == "A/0db/g1/s2/field-kuramoto-torus-random-lam0.3-clamp1"
    wide = rn.Spec("size", "recognition", "envelope", 0.0, 1.0, 2, Arm("bank", channels=8))
    assert plan.paper02_group(wide) == "tier1-recognition-envelope"      # paper 02's width-matched bank
    d = rn.Spec("design", "recognition", "envelope", 5.0, 2.0, 0, Arm("field", physics="harmonic2", boundary="helix"))
    assert plan.paper02_group(d) == "tier2-recognition-envelope-harmonic2"
    for arm in (Arm("field", channels=8), Arm("field", grid=32), Arm("field", severed=True, physics="winfree"),
                Arm("bank", channels=2), Arm("ann", arch="gru", channels=8)):
        assert plan.paper02_group(rn.Spec("size", "recognition", "envelope", 0.0, 1.0, 0, arm)) is None
    q = rn.Spec("design-quadrature", "recognition", "quadrature", 0.0, 1.0, 0, Arm("field", physics="winfree"))
    assert plan.paper02_group(q) == "tier3-recognition-quadrature"
    assert plan.paper02_group(rn.Spec("design-quadrature", "recognition", "quadrature", 0.0, 1.0, 0,
                                      Arm("field", physics="winfree", boundary="cube"))) is None


def test_the_largest_arms_are_streamed_and_everything_paper_02_ran_is_not():
    specs = plan.planned(["size"])
    assert not any(s.streamed for s in specs if plan.reused(s))
    assert all(s.streamed for s in specs if s.arm.kind != "floor" and s.arm.states > 4096)


def test_a_stage_selects_lattices_and_channels_without_changing_a_spec():
    small = plan.planned(["size"], grids=(8, 16, 32))
    assert {s.arm.grid for s in small} == {8, 16, 32}
    full = {(s.group(), s.run_id()) for s in plan.planned(["size"])}
    assert {(s.group(), s.run_id()) for s in small} <= full


def test_the_estimate_covers_every_tier_and_lattice():
    rows = plan.estimate()
    assert {r["tier"] for r in rows} == set(plan.TIERS)
    assert sum(r["runs"] for r in rows) == sum(COUNTS.values())
    assert all(r["cpu_hours"] >= 0 and r["peak_gb"] > 0 for r in rows)

"""One confirmatory run, end to end, on a small synthetic bank.

The bank has the real one's shape and speaker split but one repetition per
speaker and digit, each a short tone whose pitch names the digit, so a run is
fast and the readout has something to find.
"""

import math

import pytest
import torch

from harness.confirm import arms as am
from harness.confirm import protocol as pr
from harness.confirm import readout as ro
from harness.confirm import run as rn

LENGTH = 16000


@pytest.fixture(scope="module")
def bank():
    speakers = torch.arange(1, 61).repeat_interleave(10)
    labels = torch.arange(10).repeat(60)
    g = torch.Generator().manual_seed(0)
    lens = torch.randint(6000, 12000, (600,), generator=g)
    t = torch.arange(LENGTH) / 16000.0
    waves = torch.zeros(600, LENGTH, dtype=torch.int16)
    for i in range(600):
        f = 300.0 + 120.0 * labels[i].item() + 10.0 * speakers[i].item() % 40
        tone = 0.4 * torch.sin(2 * math.pi * f * t[:lens[i]]) + 0.02 * torch.randn(int(lens[i]), generator=g)
        waves[i, :lens[i]] = (tone * 32767).round().to(torch.int16)
    return {"waves": waves, "lens": lens, "labels": labels, "speakers": speakers,
            "reps": torch.zeros(600, dtype=torch.long), "sr": 16000, "version": 2}


@pytest.fixture
def small_order(monkeypatch):
    monkeypatch.setattr(pr, "ORDER_TRAIN", 96)
    monkeypatch.setattr(pr, "ORDER_TEST", 64)


FIELD = am.Arm("field")
SIZES = (128, 256)


def spec(arm, **kw):
    base = dict(tier="tier1", task="recognition", drive="envelope", noise_db=0.0,
                gain=2.0 if arm.uses_gain else None, seed=0, arm=arm, sizes=SIZES,
                widths=(64, 256), native_sizes=(128,))
    return rn.Spec(**{**base, **kw})


def test_a_run_is_named_by_what_decides_its_numbers():
    a = spec(FIELD)
    assert a.group() == "tier1-recognition-envelope"
    assert a.run_id() == "A/0db/g2/s0/field-kuramoto-torus-random-lam0.3-clamp1"
    assert spec(am.Arm("floor")).run_id() == "A/0db/s0/floor"                 # the floor has no gain
    assert spec(am.Arm("ann", arch="gru"), sizes=(128,)).run_id().endswith("ann-gru/n128")
    assert spec(FIELD, tier="tier2").group() == "tier2-recognition-envelope-kuramoto"
    assert spec(FIELD, noise_db=None).run_id().startswith("A/clean/")
    ids = {spec(FIELD, seed=s, gain=g).run_id() for s in (0, 1) for g in (1.0, 2.0)}
    assert len(ids) == 4


@pytest.mark.parametrize("arm", [am.Arm("floor"), FIELD, am.Arm("field", severed=True), am.Arm("bank")],
                         ids=lambda a: a.label())
def test_a_frozen_arm_is_read_at_every_size_and_width(bank, arm):
    rec = rn.execute(spec(arm), bank=bank)
    reads = am.reads(arm, "recognition")
    assert set(rec["native_widths"]) == set(reads)
    per_read = len(SIZES) * 2 + 1            # two widths at each size, plus native at 128
    assert len(rec["cells"]) == len(reads) * per_read
    assert all(0.0 <= c["acc"] <= 1.0 and "correct" in c for c in rec["cells"])
    assert rec["n_test"] == 120 and rec["env"]["torch"] == torch.__version__


def test_the_tones_are_learnable_so_the_readout_is_really_reading(bank):
    rec = rn.execute(spec(am.Arm("floor"), noise_db=None), bank=bank)
    best = max(c["acc"] for c in rec["cells"])
    assert best > 0.5


def test_with_no_drive_a_frozen_arm_carries_nothing_and_reads_chance(bank):
    # the g = 0 sanity cell: the balanced test set makes chance exactly 1 in 10
    for arm in (FIELD, am.Arm("bank")):
        rec = rn.execute(spec(arm, gain=0.0), bank=bank)
        assert all(c["acc"] == pytest.approx(0.1) for c in rec["cells"])


def test_a_per_clip_window_hands_an_undriven_field_the_clip_lengths(bank):
    # why the window is fixed: read over each clip's own length, a field with no
    # input still differs clip to clip, because it keeps rotating
    x = torch.zeros(3, 61, 16)
    tvalid = torch.tensor([30, 45, 61])
    field = am.build_frozen(FIELD, 0.0, seed=0)
    with torch.no_grad():
        sig = am.frozen_signals(FIELD, field, x)
        fixed = am.frozen_features(FIELD, sig, tvalid, "recognition", "fixed")["windowed"]
        clip = am.frozen_features(FIELD, sig, tvalid, "recognition", "clip")["windowed"]
    assert torch.equal(fixed[0], fixed[1]) and torch.equal(fixed[1], fixed[2])
    assert not torch.allclose(clip[0], clip[1]) and not torch.allclose(clip[1], clip[2])
    assert spec(FIELD, span="clip").run_id().endswith("/clipspan")


def test_severing_zeroes_the_coupling_and_nothing_else(bank):
    field = am.build_frozen(FIELD, 2.0, seed=3)
    severed = am.build_frozen(am.Arm("field", severed=True), 2.0, seed=3)
    from harness.models.field import physics_block
    a, b = physics_block(field.core), physics_block(severed.core)
    assert torch.equal(a.natural_freqs, b.natural_freqs) and not b.kernel.any() and a.kernel.any()
    assert am.meta(am.Arm("field", severed=True), severed)["effective_params"] == 1024


def test_the_field_matches_the_exploratory_runners_draw_for_the_same_seed():
    from argparse import Namespace
    from harness.runner import build_field
    args = Namespace(channels=4, grid=16, damping=0.3, clamp=1.0, substeps=1, dt=0.1, gain=2.0,
                     blocks=1, kernel_support=0, core="phase", boundary="torus", damping_learnable=False,
                     sakaguchi_alpha=0.0, harmonic2_beta=0.0, graph_k=1)
    old = build_field(args, "kuramoto", 10, seed=0).state_dict()
    new = am.build_frozen(FIELD, 2.0, seed=0).state_dict()
    assert all(torch.equal(old[k], new[k]) for k in ("core.blocks.0.kernel", "core.blocks.0.natural_freqs"))


def test_a_trained_network_is_read_at_its_own_training_size(bank):
    rec = rn.execute(spec(am.Arm("ann", arch="gru"), sizes=(128,), native_sizes=(128,)), bank=bank)
    assert {c["n_train"] for c in rec["cells"]} == {128}
    assert rec["health"]["epochs"] == am.EPOCHS and 0.0 <= rec["head_acc"] <= 1.0
    assert rec["arm_meta"]["trained_params"] > 0


def test_a_smaller_run_reproduces_the_same_cell_of_a_larger_one(bank):
    # the tier 2 cell of a configuration must equal its tier 1 cell: same clips, same order
    big = rn.execute(spec(am.Arm("floor")), bank=bank)
    small = rn.execute(spec(am.Arm("floor"), tier="tier2", sizes=(128,)), bank=bank)
    key = lambda c: (c["read"], c["n_train"], c["width"])  # noqa: E731
    big_cells = {key(c): c for c in big["cells"]}
    for c in small["cells"]:
        assert c["acc"] == big_cells[key(c)]["acc"] and c["correct"] == big_cells[key(c)]["correct"]


def test_primary_bits_mode_keeps_bits_only_for_the_primary_read_at_the_primary_size(bank, monkeypatch):
    monkeypatch.setattr(rn, "PRIMARY_SIZE", 128)
    rec = rn.execute(spec(am.Arm("floor"), bits="primary"), bank=bank)
    for c in rec["cells"]:
        assert ("correct" in c) == (c["n_train"] == 128 and c["read"].split("@")[0] == "windowed")


def test_the_order_task_runs_on_balanced_sets_and_reads_only_the_whole_span(bank, small_order):
    rec = rn.execute(spec(FIELD, task="order", pair=(3, 7), sizes=(96,), native_sizes=(96,)), bank=bank)
    assert {c["read"] for c in rec["cells"]} == {"pooled", "pooled+rate"}
    assert rec["n_test"] == 64
    first, second, labels = pr.order_set(bank, pr.protocol_a(bank)[1], (3, 7), 64, 0)
    assert labels.sum().item() == 32 and torch.equal(bank["labels"][first[labels == 0]], torch.full((32,), 3))


def test_the_becker_protocol_chooses_the_penalty_on_its_own_validation_fold(bank):
    rec = rn.execute(spec(am.Arm("floor"), protocol="B", fold=2, noise_db=None, sizes=(360,),
                          native_sizes=()), bank=bank)
    assert rec["n_test"] == 120 and {c["n_train"] for c in rec["cells"]} == {360}


def test_a_run_is_recorded_once_and_found_again(bank, tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    s = spec(am.Arm("floor"))
    rn.write(s, rn.execute(s, bank=bank))
    rn.write(s, rn.execute(s, bank=bank))
    assert rn.recorded_ids(s.group()) == {s.run_id()}
    rec = rn.load_group(s.group())["runs"][s.run_id()]
    c = rec["cells"][0]
    assert ro.unpack(c["correct"], rec["n_test"]).double().mean().item() == pytest.approx(c["acc"])


def test_cached_rows_are_exactly_the_rows_a_run_would_compute(bank, tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "ROWS_DIR", tmp_path)
    s = spec(am.Arm("floor"), noise_db=5.0)
    without = rn.execute(s, bank=bank)
    pr.build_rows(bank, "envelope", 5.0)
    assert pr.load_rows(pr.rows_path("envelope", 5.0), len(bank["labels"])) is not None
    with_cache = rn.execute(s, bank=bank)
    assert [c["acc"] for c in with_cache["cells"]] == [c["acc"] for c in without["cells"]]
    # a cache built for another bank is ignored rather than trusted
    assert pr.load_rows(pr.rows_path("envelope", 5.0), len(bank["labels"]) + 1) is None


def test_cached_order_sets_match_the_sets_built_on_the_fly(bank, tmp_path, monkeypatch, small_order):
    monkeypatch.setattr(pr, "ROWS_DIR", tmp_path)
    s = spec(FIELD, task="order", pair=(1, 8), sizes=(96,), native_sizes=(96,), noise_db=0.0)
    without = rn.execute(s, bank=bank)
    for code in (0, 1):
        pr.build_order_rows(bank, (1, 8), code, 0.0)
    with_cache = rn.execute(s, bank=bank)
    assert [c["acc"] for c in with_cache["cells"]] == [c["acc"] for c in without["cells"]]


def test_a_reads_filter_records_only_those_reads(bank):
    rec = rn.execute(spec(FIELD, reads=("windowed", "windowed+rate")), bank=bank)
    assert {c["read"] for c in rec["cells"]} == {"windowed", "windowed+rate"}

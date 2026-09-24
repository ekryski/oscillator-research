"""One run, end to end, on a small synthetic bank.

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


FIELD = am.Arm("field")
SIZES = (128, 256)


def spec(arm, **kw):
    base = dict(tier="size", task="recognition", drive="envelope", noise_db=0.0,
                gain=2.0 if arm.uses_gain else None, seed=0, arm=arm, sizes=SIZES,
                widths=(64, 256), native_sizes=(128,), bits="all")
    return rn.Spec(**{**base, **kw})


def test_a_run_is_named_by_what_decides_its_numbers_in_paper_02s_form():
    a = spec(FIELD)
    assert a.group() == "size-envelope-16x16"
    assert a.run_id() == "A/0db/g2/s0/field-kuramoto-torus-random-lam0.3-clamp1"
    assert spec(am.Arm("floor")).run_id() == "A/0db/s0/floor"                 # the floor has no gain
    assert spec(am.Arm("ann", arch="gru"), sizes=(128,)).run_id().endswith("ann-gru/n128")
    assert spec(am.Arm("field", physics="winfree", grid=64), tier="design").group() == "design-envelope-64x64-winfree"
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


def test_a_trained_network_is_read_at_its_own_training_size(bank):
    rec = rn.execute(spec(am.Arm("ann", arch="gru"), sizes=(128,), native_sizes=(128,)), bank=bank)
    assert {c["n_train"] for c in rec["cells"]} == {128}
    assert rec["health"]["epochs"] == am.EPOCHS and 0.0 <= rec["head_acc"] <= 1.0
    assert rec["arm_meta"]["trained_params"] > 0


def test_a_smaller_run_reproduces_the_same_cell_of_a_larger_one(bank):
    # paper 02's tier 1 ran three training sizes; its 2,048-clip cell is the cell a paper 03 run computes
    big = rn.execute(spec(FIELD), bank=bank)
    small = rn.execute(spec(FIELD, tier="design", sizes=(128,)), bank=bank)
    key = lambda c: (c["read"], c["n_train"], c["width"])  # noqa: E731
    big_cells = {key(c): c for c in big["cells"]}
    for c in small["cells"]:
        assert c["acc"] == big_cells[key(c)]["acc"] and c["correct"] == big_cells[key(c)]["correct"]


def test_primary_bits_mode_keeps_bits_only_for_the_primary_read_at_the_primary_size(bank, monkeypatch):
    monkeypatch.setattr(rn, "PRIMARY_SIZE", 128)
    rec = rn.execute(spec(am.Arm("floor"), bits="primary"), bank=bank)
    for c in rec["cells"]:
        assert ("correct" in c) == (c["n_train"] == 128 and c["read"].split("@")[0] == "windowed")


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


def test_a_reads_filter_records_only_those_reads(bank):
    rec = rn.execute(spec(FIELD, reads=("windowed", "windowed+rate")), bank=bank)
    assert {c["read"] for c in rec["cells"]} == {"windowed", "windowed+rate"}


def test_adding_the_lattice_left_every_registered_label_as_it_was():
    assert FIELD.label() == "field-kuramoto-torus-random-lam0.3-clamp1"
    assert am.Arm("field", channels=16).label() == "field-kuramoto-torus-random-lam0.3-clamp1-c16"
    assert (am.Arm("floor").label(), am.Arm("bank", channels=8).label()) == ("floor", "bank-c8")
    assert am.Arm("field", grid=16, bands=16).label() == FIELD.label()       # a band per row is the default
    assert am.Arm("bank", channels=2, grid=32, bands=16).label() == "bank-c2-32x32-16bands"
    assert pr.rows_path("envelope", 0.0).name == "envelope-0db.pt"


def test_bands_map_onto_rows_in_frequency_order_without_anything_fitted():
    rows = torch.arange(16.0).view(1, 1, 16)
    assert pr.to_rows(rows, 16) is rows
    assert pr.to_rows(rows, 32)[0, 0, :4].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert pr.to_rows(rows, 8)[0, 0, :3].tolist() == [0.5, 2.5, 4.5]
    with pytest.raises(ValueError, match="cannot map"):
        pr.to_rows(rows, 12)


@pytest.mark.parametrize("arm", [am.Arm("field", channels=1, grid=8), am.Arm("bank", channels=2, grid=8, bands=16),
                                 am.Arm("field", channels=1, grid=32, bands=16), am.Arm("floor", grid=32, bands=16)],
                         ids=lambda a: a.label())
def test_a_lattice_of_another_size_runs_with_its_rows(bank, arm):
    rec = rn.execute(spec(arm, reads=("windowed",)), bank=bank)
    signals = {"field": 2 * arm.states, "bank": arm.states, "floor": arm.grid}[arm.kind]
    assert rec["native_widths"] == {"windowed": 12 * signals}
    assert rec["arm_meta"]["states"] == (arm.states if arm.kind != "floor" else 0)
    assert all(0.0 <= c["acc"] <= 1.0 for c in rec["cells"])


def test_paper_02s_labels_are_kept_and_paper_03s_sizes_extend_them():
    assert am.Arm("ann", arch="gru").label() == "ann-gru"
    assert am.Arm("ann", arch="s4d", channels=8, grid=32, bands=16).label() == "ann-s4d-c8-32x32-16bands"
    assert am.Arm("field", channels=1, grid=128).label() == "field-kuramoto-torus-random-lam0.3-clamp1-c1-128x128"
    assert am.Arm("field", channels=16, grid=8).budget == 2 * 16 * 8 * 8


def test_quadrature_rows_map_onto_rows_along_the_band_axis():
    rows = torch.arange(32.0).view(1, 1, 16, 2)
    assert pr.to_rows(rows, 32).shape == (1, 1, 32, 2)
    assert pr.to_rows(rows, 8)[0, 0, 0].tolist() == [1.0, 2.0]          # the mean of bands 0 and 1


@pytest.mark.parametrize("drive", ["quadrature", "carrier"])
def test_the_other_pathways_run_on_another_lattice(bank, drive, monkeypatch):
    monkeypatch.setitem(rn.BATCH, "carrier", 64)
    arm = am.Arm("field", channels=1, grid=8, bands=16)
    rec = rn.execute(spec(arm, drive=drive, gain=1.0, reads=("windowed",), sizes=(128,)), bank=bank)
    assert all(0.0 <= c["acc"] <= 1.0 for c in rec["cells"])


def test_a_trained_baseline_is_sized_to_its_network_and_driven_by_its_rows(bank):
    arm = am.Arm("ann", arch="gru", channels=1, grid=8)
    rec = rn.execute(spec(arm, sizes=(128,), native_sizes=(128,)), bank=bank)
    assert rec["arm_meta"]["budget"] == 128 and rec["arm_meta"]["trained_params"] <= 128
    assert rec["arm_meta"]["width"] == am.ann_width("gru", 8, 128)

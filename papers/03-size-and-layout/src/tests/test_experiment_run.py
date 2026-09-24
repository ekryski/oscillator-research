"""One run, end to end, on a small synthetic bank.

The bank has the real one's shape and speaker split but one repetition per
speaker and digit, each a short tone whose pitch names the digit, so a run is
fast and the readout has something to find.
"""

import math

import pytest
import torch

from harness.experiment import arms as am
from harness.experiment import protocol as pr
from harness.experiment import readout as ro
from harness.experiment import run as rn

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


FIELD = am.Arm("network")
SIZES = (128, 256)


def spec(arm, **kw):
    base = dict(tier="size", task="recognition", pathway="spectrogram", noise_db=0.0,
                gain=2.0 if arm.uses_gain else None, seed=0, arm=arm, sizes=SIZES,
                widths=(64, 256), native_sizes=(128,), bits="all")
    return rn.Spec(**{**base, **kw})


def test_a_run_is_named_by_what_decides_its_numbers_in_paper_02s_form():
    a = spec(FIELD)
    assert a.group() == "size-recognition-spectrogram-16x16"
    assert a.run_id() == "A/0db/g2/s0/coupled-kuramoto-torus-random-restoring0.3-ceiling1"
    assert spec(am.Arm("baseline")).run_id() == "A/0db/s0/baseline"              # the baseline has no gain
    assert spec(am.Arm("trained", arch="gru"), sizes=(128,)).run_id().endswith("trained-gru/n128")
    assert spec(am.Arm("network", coupling="winfree", grid=64), tier="design").group() == "design-recognition-spectrogram-64x64-winfree"
    assert spec(FIELD, noise_db=None).run_id().startswith("A/clean/")
    ids = {spec(FIELD, seed=s, gain=g).run_id() for s in (0, 1) for g in (1.0, 2.0)}
    assert len(ids) == 4


@pytest.mark.parametrize("arm", [am.Arm("baseline"), FIELD, am.Arm("network", coupled=False), am.Arm("bank")],
                         ids=lambda a: a.label())
def test_an_untrained_arm_is_read_at_every_size_and_width(bank, arm):
    rec = rn.execute(spec(arm), bank=bank)
    reads = am.reads(arm, "recognition")
    assert set(rec["native_widths"]) == set(reads)
    expected = 0
    for read in reads:
        native = rec["native_widths"][read]
        below = sum(w < native for w in (64, 256))
        # each width below the arm's own under both projections, the rest unprojected, and native at 128
        expected += len(SIZES) * (2 * below + (2 - below)) + 1
    assert len(rec["cells"]) == expected
    assert all(0.0 <= c["acc"] <= 1.0 and "correct" in c for c in rec["cells"])
    for c in rec["cells"]:
        assert c["projection"] == ("none" if c["effective_width"] >= rec["native_widths"][c["read"]] else
                                   c["projection"]) and c["projection"] in ("fixed", "seeded", "none")
    assert rec["n_test"] == 120 and rec["env"]["torch"] == torch.__version__


def test_the_tones_are_learnable_so_the_readout_is_really_reading(bank):
    rec = rn.execute(spec(am.Arm("baseline"), noise_db=None), bank=bank)
    best = max(c["acc"] for c in rec["cells"])
    assert best > 0.5


def test_with_no_input_an_untrained_arm_carries_nothing_and_reads_chance(bank):
    # the g = 0 sanity cell: the balanced test set makes chance exactly 1 in 10
    for arm in (FIELD, am.Arm("bank")):
        rec = rn.execute(spec(arm, gain=0.0), bank=bank)
        assert all(c["acc"] == pytest.approx(0.1) for c in rec["cells"])


def test_a_per_clip_window_hands_an_undriven_network_the_clip_lengths(bank):
    # why the window is fixed: read over each clip's own length, a field with no
    # input still differs clip to clip, because it keeps rotating
    x = torch.zeros(3, 61, 16)
    tvalid = torch.tensor([30, 45, 61])
    field = am.build_untrained(FIELD, 0.0, seed=0)
    with torch.no_grad():
        sig = am.untrained_signals(FIELD, field, x)
        fixed = am.untrained_features(FIELD, sig, tvalid, "recognition", "fixed")["windowed"]
        clip = am.untrained_features(FIELD, sig, tvalid, "recognition", "clip")["windowed"]
    assert torch.equal(fixed[0], fixed[1]) and torch.equal(fixed[1], fixed[2])
    assert not torch.allclose(clip[0], clip[1]) and not torch.allclose(clip[1], clip[2])
    assert spec(FIELD, span="clip").run_id().endswith("/clipspan")


def test_severing_zeroes_the_coupling_and_nothing_else(bank):
    field = am.build_untrained(FIELD, 2.0, seed=3)
    uncoupled = am.build_untrained(am.Arm("network", coupled=False), 2.0, seed=3)
    from harness.models.field import physics_block
    a, b = physics_block(field.core), physics_block(uncoupled.core)
    assert torch.equal(a.natural_freqs, b.natural_freqs) and not b.kernel.any() and a.kernel.any()
    assert am.meta(am.Arm("network", coupled=False), uncoupled)["effective_params"] == 1024


def test_every_projected_width_is_read_under_the_fixed_and_the_seeded_projection(bank):
    cells = {}
    for seed in (0, 1):
        rec = rn.execute(spec(FIELD, seed=seed, sizes=(128,), native_sizes=(), reads=("windowed",)), bank=bank)
        cells[seed] = {(c["width"], c["projection"]): c for c in rec["cells"]}
        assert set(cells[seed]) == {(w, p) for w in (64, 256) for p in ("fixed", "seeded")}
        assert all("correct" in c for c in rec["cells"])
    # the seeded projection is drawn from 4242 + 1 + seed, the fixed one from 4242 for every seed
    from harness.measurement.features import projection_matrix
    assert not torch.equal(projection_matrix(1000, 64, 0), projection_matrix(1000, 64, 1))
    g = torch.Generator().manual_seed(4242 + 1 + 1)
    assert torch.equal(projection_matrix(1000, 64, 1), (torch.randn(1000, 4096, generator=g) / math.sqrt(1000))[:, :64])
    g = torch.Generator().manual_seed(4242)
    assert torch.equal(projection_matrix(1000, 64), (torch.randn(1000, 4096, generator=g) / math.sqrt(1000))[:, :64])


def test_a_trained_network_is_read_at_its_own_training_size(bank):
    rec = rn.execute(spec(am.Arm("trained", arch="gru"), sizes=(128,), native_sizes=(128,)), bank=bank)
    assert {c["n_train"] for c in rec["cells"]} == {128}
    assert rec["health"]["epochs"] == am.EPOCHS and 0.0 <= rec["head_acc"] <= 1.0
    assert rec["arm_meta"]["trained_params"] > 0


def test_a_smaller_run_reproduces_the_same_cell_of_a_larger_one(bank):
    # paper 02's tier 1 ran three training sizes; its 2,048-clip cell is the cell a paper 03 run computes
    big = rn.execute(spec(FIELD), bank=bank)
    small = rn.execute(spec(FIELD, tier="design", sizes=(128,)), bank=bank)
    key = lambda c: (c["read"], c["n_train"], c["width"], c["projection"])  # noqa: E731
    big_cells = {key(c): c for c in big["cells"]}
    for c in small["cells"]:
        assert c["acc"] == big_cells[key(c)]["acc"] and c["correct"] == big_cells[key(c)]["correct"]


def test_primary_bits_mode_keeps_bits_only_for_the_primary_read_at_the_primary_size(bank, monkeypatch):
    monkeypatch.setattr(rn, "PRIMARY_SIZE", 128)
    rec = rn.execute(spec(am.Arm("baseline"), bits="primary"), bank=bank)
    for c in rec["cells"]:
        assert ("correct" in c) == (c["n_train"] == 128 and c["read"].split("@")[0] == "windowed")


def test_a_run_is_recorded_once_and_found_again(bank, tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path))
    s = spec(am.Arm("baseline"))
    rn.write(s, rn.execute(s, bank=bank))
    rn.write(s, rn.execute(s, bank=bank))
    assert rn.recorded_ids(s.group()) == {s.run_id()}
    rec = rn.load_group(s.group())["runs"][s.run_id()]
    c = rec["cells"][0]
    assert ro.unpack(c["correct"], rec["n_test"]).double().mean().item() == pytest.approx(c["acc"])


def test_cached_rows_are_exactly_the_rows_a_run_would_compute(bank, tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "ROWS_DIR", tmp_path)
    s = spec(am.Arm("baseline"), noise_db=5.0)
    without = rn.execute(s, bank=bank)
    pr.build_rows(bank, "spectrogram", 5.0)
    assert pr.load_rows(pr.rows_path("spectrogram", 5.0), len(bank["labels"])) is not None
    with_cache = rn.execute(s, bank=bank)
    assert [c["acc"] for c in with_cache["cells"]] == [c["acc"] for c in without["cells"]]
    # a cache built for another bank is ignored rather than trusted
    assert pr.load_rows(pr.rows_path("spectrogram", 5.0), len(bank["labels"]) + 1) is None


def test_a_reads_filter_records_only_those_reads(bank):
    rec = rn.execute(spec(FIELD, reads=("windowed", "windowed+rate")), bank=bank)
    assert {c["read"] for c in rec["cells"]} == {"windowed", "windowed+rate"}


def test_adding_the_lattice_and_the_window_left_every_paper_02_label_as_it_was():
    assert FIELD.label() == "coupled-kuramoto-torus-random-restoring0.3-ceiling1"
    assert am.Arm("network", channels=16).label() == "coupled-kuramoto-torus-random-restoring0.3-ceiling1-ch16"
    assert (am.Arm("baseline").label(), am.Arm("bank", channels=8).label()) == ("baseline", "bank-width")
    assert am.Arm("network", grid=16, bands=16).label() == FIELD.label()       # a band per row is the default
    assert am.Arm("bank", channels=2, grid=32, bands=16).label() == "bank-ch2-32x32-16bands"
    assert am.Arm("network", window=512).label() == FIELD.label()               # paper 02's window is the default
    assert am.Arm("bank", channels=2, grid=64, window=1024).label() == "bank-ch2-64x64-w1024"
    assert pr.rows_path("spectrogram", 0.0).name == "spectrogram-0db.pt"


def test_bands_map_onto_rows_in_frequency_order_without_anything_fitted():
    rows = torch.arange(16.0).view(1, 1, 16)
    assert pr.to_rows(rows, 16) is rows
    assert pr.to_rows(rows, 32)[0, 0, :4].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert pr.to_rows(rows, 8)[0, 0, :3].tolist() == [0.5, 2.5, 4.5]
    with pytest.raises(ValueError, match="cannot map"):
        pr.to_rows(rows, 12)


@pytest.mark.parametrize("arm", [am.Arm("network", channels=1, grid=8), am.Arm("bank", channels=2, grid=8, bands=16),
                                 am.Arm("network", channels=1, grid=32, bands=16), am.Arm("baseline", grid=32, bands=16)],
                         ids=lambda a: a.label())
def test_a_lattice_of_another_size_runs_with_its_rows(bank, arm):
    rec = rn.execute(spec(arm, reads=("windowed",)), bank=bank)
    signals = {"network": 2 * arm.states, "bank": arm.states, "baseline": arm.grid}[arm.kind]
    assert rec["native_widths"] == {"windowed": 12 * signals}
    assert rec["arm_meta"]["states"] == (arm.states if arm.kind != "baseline" else 0)
    assert all(0.0 <= c["acc"] <= 1.0 for c in rec["cells"])


def test_paper_02s_labels_are_kept_and_paper_03s_sizes_extend_them():
    assert am.Arm("trained", arch="gru").label() == "trained-gru"
    assert am.Arm("trained", arch="s4d", channels=8, grid=32, bands=16).label() == "trained-s4d-ch8-32x32-16bands"
    assert am.Arm("network", channels=1, grid=128).label() == "coupled-kuramoto-torus-random-restoring0.3-ceiling1-ch1-128x128"
    assert am.Arm("network", channels=16, grid=8).budget == 2 * 16 * 8 * 8


def test_quadrature_rows_map_onto_rows_along_the_band_axis():
    rows = torch.arange(32.0).view(1, 1, 16, 2)
    assert pr.to_rows(rows, 32).shape == (1, 1, 32, 2)
    assert pr.to_rows(rows, 8)[0, 0, 0].tolist() == [1.0, 2.0]          # the mean of bands 0 and 1


@pytest.mark.parametrize("pathway", ["quadrature", "carrier"])
def test_the_other_pathways_run_on_another_lattice(bank, pathway, monkeypatch):
    monkeypatch.setitem(rn.BATCH, "carrier", 64)
    arm = am.Arm("network", channels=1, grid=8, bands=16)
    rec = rn.execute(spec(arm, pathway=pathway, gain=1.0, reads=("windowed",), sizes=(128,)), bank=bank)
    assert all(0.0 <= c["acc"] <= 1.0 for c in rec["cells"])


def test_a_trained_baseline_is_sized_to_its_network_and_driven_by_its_rows(bank):
    arm = am.Arm("trained", arch="gru", channels=1, grid=8)
    rec = rn.execute(spec(arm, sizes=(128,), native_sizes=(128,)), bank=bank)
    assert rec["arm_meta"]["budget"] == 128 and rec["arm_meta"]["trained_params"] <= 128
    assert rec["arm_meta"]["width"] == am.trained_width("gru", 8, 128)


def test_a_network_records_how_synchronized_and_how_locked_it_is_and_nothing_else_does(bank):
    for arm in (FIELD, am.Arm("network", coupled=False), am.Arm("network", coupling="stuart-landau")):
        inst = rn.execute(spec(arm, reads=("windowed",)), bank=bank)["instruments"]
        assert set(inst) == {"R", "plv", "entrained", "amplitude"}
        for name in ("R", "plv", "entrained"):
            assert 0.0 <= inst[name]["mean"] <= 1.0 and len(inst[name]["by_class"]) == 10
        if arm.coupling != "stuart-landau":
            assert inst["amplitude"]["mean"] == pytest.approx(1.0, abs=1e-4)   # a phase core stays on the circle
    assert "instruments" not in rn.execute(spec(am.Arm("bank"), reads=("windowed",)), bank=bank)


def test_the_instruments_leave_every_cell_as_it_was(bank, monkeypatch):
    cells = rn.execute(spec(FIELD), bank=bank)["cells"]
    monkeypatch.setattr(am, "network_instruments", lambda *a, **k: {})
    assert cells == rn.execute(spec(FIELD), bank=bank)["cells"]


def test_the_order_parameter_reads_one_for_a_locked_channel_and_near_zero_for_a_spread_one():
    g, t = 4, 20
    theta = torch.full((1, t, 1, g, g), 0.3)                  # every oscillator at one phase
    sig = torch.cat((theta.sin().flatten(2), theta.cos().flatten(2)), dim=2)
    rows = torch.rand(1, t, g)
    assert am.network_instruments(sig, rows, "spectrogram", 1, g, lo=0)["R"].item() == pytest.approx(1.0)
    spread = torch.linspace(0, 2 * math.pi, g * g + 1)[:-1].view(1, 1, 1, g, g).expand(1, t, 1, g, g)
    sig = torch.cat((spread.sin().flatten(2), spread.cos().flatten(2)), dim=2)
    assert am.network_instruments(sig, rows, "spectrogram", 1, g, lo=0)["R"].item() < 1e-5

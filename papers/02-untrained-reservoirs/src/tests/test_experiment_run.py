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


@pytest.fixture
def small_order(monkeypatch):
    monkeypatch.setattr(pr, "ORDER_TRAIN", 96)
    monkeypatch.setattr(pr, "ORDER_TEST", 64)


COUPLED = am.Arm("network")
SIZES = (128, 256)


def spec(arm, **kw):
    base = dict(tier="tier1", task="recognition", pathway="spectrogram", noise_db=0.0,
                gain=2.0 if arm.uses_gain else None, seed=0, arm=arm, sizes=SIZES,
                widths=(64, 256), native_sizes=(128,))
    return rn.Spec(**{**base, **kw})


def test_a_run_is_named_by_what_decides_its_numbers():
    a = spec(COUPLED)
    assert a.group() == "tier1-recognition-spectrogram"
    assert a.run_id() == "A/0db/g2/s0/coupled-kuramoto-torus-random-restoring0.3-ceiling1"
    assert spec(am.Arm("baseline")).run_id() == "A/0db/s0/baseline"           # the baseline has no gain
    assert spec(am.Arm("trained", arch="gru"), sizes=(128,)).run_id().endswith("trained-gru/n128")
    assert spec(COUPLED, tier="tier2").group() == "tier2-recognition-spectrogram-kuramoto"
    assert spec(COUPLED, noise_db=None).run_id().startswith("A/clean/")
    ids = {spec(COUPLED, seed=s, gain=g).run_id() for s in (0, 1) for g in (1.0, 2.0)}
    assert len(ids) == 4


@pytest.mark.parametrize("arm", [am.Arm("baseline"), COUPLED, am.Arm("network", coupled=False), am.Arm("bank")],
                         ids=lambda a: a.label())
def test_an_untrained_arm_is_read_at_every_size_and_width(bank, arm):
    rec = rn.execute(spec(arm), bank=bank)
    reads = am.reads(arm, "recognition")
    assert set(rec["native_widths"]) == set(reads)
    per_read = len(SIZES) * 2 + 1            # two widths at each size, plus native at 128
    assert len(rec["cells"]) == len(reads) * per_read
    assert all(0.0 <= c["acc"] <= 1.0 and "correct" in c for c in rec["cells"])
    assert rec["n_test"] == 120 and rec["env"]["torch"] == torch.__version__


def test_the_tones_are_learnable_so_the_readout_is_really_reading(bank):
    rec = rn.execute(spec(am.Arm("baseline"), noise_db=None), bank=bank)
    best = max(c["acc"] for c in rec["cells"])
    assert best > 0.5


def test_with_no_input_an_untrained_arm_carries_nothing_and_reads_chance(bank):
    # the g = 0 sanity cell: the balanced test set makes chance exactly 1 in 10
    for arm in (COUPLED, am.Arm("bank")):
        rec = rn.execute(spec(arm, gain=0.0), bank=bank)
        assert all(c["acc"] == pytest.approx(0.1) for c in rec["cells"])


def test_a_per_clip_window_hands_an_undriven_network_the_clip_lengths(bank):
    # why the window is fixed: read over each clip's own length, a network with no
    # input still differs clip to clip, because it keeps rotating
    x = torch.zeros(3, 61, 16)
    tvalid = torch.tensor([30, 45, 61])
    network = am.build_untrained(COUPLED, 0.0, seed=0)
    with torch.no_grad():
        sig = am.untrained_signals(COUPLED, network, x)
        fixed = am.untrained_features(COUPLED, sig, tvalid, "recognition", "fixed")["windowed"]
        clip = am.untrained_features(COUPLED, sig, tvalid, "recognition", "clip")["windowed"]
    assert torch.equal(fixed[0], fixed[1]) and torch.equal(fixed[1], fixed[2])
    assert not torch.allclose(clip[0], clip[1]) and not torch.allclose(clip[1], clip[2])
    assert spec(COUPLED, span="clip").run_id().endswith("/clipspan")


def test_uncoupling_zeroes_the_coupling_and_nothing_else(bank):
    coupled = am.build_untrained(COUPLED, 2.0, seed=3)
    uncoupled = am.build_untrained(am.Arm("network", coupled=False), 2.0, seed=3)
    from harness.models.field import physics_block
    a, b = physics_block(coupled.core), physics_block(uncoupled.core)
    assert torch.equal(a.natural_freqs, b.natural_freqs) and not b.kernel.any() and a.kernel.any()
    assert am.meta(am.Arm("network", coupled=False), uncoupled)["effective_params"] == 1024


def test_a_trained_network_is_read_at_its_own_training_size(bank):
    rec = rn.execute(spec(am.Arm("trained", arch="gru"), sizes=(128,), native_sizes=(128,)), bank=bank)
    assert {c["n_train"] for c in rec["cells"]} == {128}
    assert rec["health"]["epochs"] == am.EPOCHS and 0.0 <= rec["head_acc"] <= 1.0
    assert rec["arm_meta"]["trained_params"] > 0


def test_a_smaller_run_reproduces_the_same_cell_of_a_larger_one(bank):
    # the tier 2 cell of a configuration must equal its tier 1 cell: same clips, same order
    big = rn.execute(spec(am.Arm("baseline")), bank=bank)
    small = rn.execute(spec(am.Arm("baseline"), tier="tier2", sizes=(128,)), bank=bank)
    key = lambda c: (c["read"], c["n_train"], c["width"])  # noqa: E731
    big_cells = {key(c): c for c in big["cells"]}
    for c in small["cells"]:
        assert c["acc"] == big_cells[key(c)]["acc"] and c["correct"] == big_cells[key(c)]["correct"]


def test_primary_bits_mode_keeps_bits_only_for_the_primary_read_at_the_primary_size(bank, monkeypatch):
    monkeypatch.setattr(rn, "PRIMARY_SIZE", 128)
    rec = rn.execute(spec(am.Arm("baseline"), bits="primary"), bank=bank)
    for c in rec["cells"]:
        assert ("correct" in c) == (c["n_train"] == 128 and c["read"].split("@")[0] == "windowed")


def test_the_order_task_runs_on_balanced_sets_and_reads_only_the_whole_span(bank, small_order):
    rec = rn.execute(spec(COUPLED, task="order", pair=(3, 7), sizes=(96,), native_sizes=(96,)), bank=bank)
    assert {c["read"] for c in rec["cells"]} == {"pooled", "pooled+rate"}
    assert rec["n_test"] == 64
    first, second, labels = pr.order_set(bank, pr.protocol_a(bank)[1], (3, 7), 64, 0)
    assert labels.sum().item() == 32 and torch.equal(bank["labels"][first[labels == 0]], torch.full((32,), 3))


def test_the_becker_protocol_chooses_the_penalty_on_its_own_validation_fold(bank):
    rec = rn.execute(spec(am.Arm("baseline"), protocol="B", fold=2, noise_db=None, sizes=(360,),
                          native_sizes=()), bank=bank)
    assert rec["n_test"] == 120 and {c["n_train"] for c in rec["cells"]} == {360}


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


def test_cached_order_sets_match_the_sets_built_on_the_fly(bank, tmp_path, monkeypatch, small_order):
    monkeypatch.setattr(pr, "ROWS_DIR", tmp_path)
    s = spec(COUPLED, task="order", pair=(1, 8), sizes=(96,), native_sizes=(96,), noise_db=0.0)
    without = rn.execute(s, bank=bank)
    for code in (0, 1):
        pr.build_order_rows(bank, (1, 8), code, 0.0)
    with_cache = rn.execute(s, bank=bank)
    assert [c["acc"] for c in with_cache["cells"]] == [c["acc"] for c in without["cells"]]


def test_a_reads_filter_records_only_those_reads(bank):
    rec = rn.execute(spec(COUPLED, reads=("windowed", "windowed+rate")), bank=bank)
    assert {c["read"] for c in rec["cells"]} == {"windowed", "windowed+rate"}


def test_both_projections_keep_the_fixed_cells_exactly_and_add_only_projected_seeded_ones(bank):
    fixed = rn.execute(spec(COUPLED, reads=("windowed",)), bank=bank)
    both = rn.execute(spec(COUPLED, reads=("windowed",), projection="both"), bank=bank)
    strip = lambda c: {k: v for k, v in c.items() if k != "projection"}  # noqa: E731
    assert [strip(c) for c in both["cells"] if c["projection"] != "seeded"] == fixed["cells"]
    assert all((c["projection"] == "none") == (c["effective_width"] == both["native_widths"]["windowed"])
               for c in both["cells"] if c["projection"] != "seeded")
    seeded = [c for c in both["cells"] if c["projection"] == "seeded"]
    native = both["native_widths"]["windowed"]
    assert seeded and all(c["effective_width"] < native for c in seeded)
    assert {(c["n_train"], c["width"]) for c in seeded} == {(n, w) for n in SIZES for w in (64, 256)}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple Silicon GPU")
@pytest.mark.parametrize("arm", [COUPLED, am.Arm("bank")], ids=lambda a: a.label())
def test_an_untrained_arm_reads_the_same_on_the_apple_silicon_gpu(bank, arm):
    assert rn.execute(spec(arm), device="mps", bank=bank)["cells"] == rn.execute(spec(arm), bank=bank)["cells"]


def test_a_network_records_how_synchronized_and_how_locked_it_is_and_nothing_else_does(bank):
    for arm in (COUPLED, am.Arm("network", coupled=False), am.Arm("network", coupling="stuart-landau")):
        inst = rn.execute(spec(arm, reads=("windowed",)), bank=bank)["instruments"]
        assert set(inst) == {"R", "plv", "entrained", "amplitude"}
        for name in ("R", "plv", "entrained"):
            assert 0.0 <= inst[name]["mean"] <= 1.0 and len(inst[name]["by_class"]) == 10
        if arm.coupling != "stuart-landau":
            assert inst["amplitude"]["mean"] == pytest.approx(1.0, abs=1e-4)   # a phase core stays on the circle
    assert "instruments" not in rn.execute(spec(am.Arm("bank"), reads=("windowed",)), bank=bank)


def test_the_instruments_leave_every_cell_as_it_was(bank):
    cells = rn.execute(spec(COUPLED), bank=bank)["cells"]
    assert cells == rn.execute(spec(COUPLED), bank=bank)["cells"]


def test_the_order_parameter_reads_one_for_a_locked_channel_and_near_zero_for_a_spread_one():
    g, t = 4, 20
    theta = torch.zeros(1, t, 1, g, g)
    theta[..., :, :] = 0.3                                   # every oscillator at one phase
    sig = torch.cat((theta.sin().flatten(2), theta.cos().flatten(2)), dim=2)
    rows = torch.rand(1, t, g)
    assert am.network_instruments(sig, rows, "spectrogram", 1, g, lo=0)["R"].item() == pytest.approx(1.0)
    spread = torch.linspace(0, 2 * math.pi, g * g + 1)[:-1].view(1, 1, 1, g, g).expand(1, t, 1, g, g)
    sig = torch.cat((spread.sin().flatten(2), spread.cos().flatten(2)), dim=2)
    assert am.network_instruments(sig, rows, "spectrogram", 1, g, lo=0)["R"].item() < 1e-5

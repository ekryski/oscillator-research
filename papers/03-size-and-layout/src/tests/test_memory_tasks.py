"""The memory tasks: paper 02's order task, and the digit-sequence task, on the small synthetic bank."""

import itertools
import math

import pytest
import torch

from harness.experiment import arms as am
from harness.experiment import protocol as pr
from harness.experiment import run as rn

from .test_experiment_run import bank, spec  # noqa: F401  (the synthetic bank fixture)


@pytest.fixture
def small_sets(monkeypatch):
    monkeypatch.setattr(pr, "ORDER_TRAIN", 96)
    monkeypatch.setattr(pr, "ORDER_TEST", 64)
    monkeypatch.setattr(pr, "SEQUENCE_TRAIN", 120)
    monkeypatch.setattr(pr, "SEQUENCE_TEST", 80)


def _paper02_order_clips(bank, first, second, pair, set_code, start, noise_db):  # noqa: F811
    """Paper 02's order_clips, verbatim but for its constants' names."""
    n = len(first)
    waves = torch.zeros(n, 4352 + 2 * 16000 + 1600)
    lens = torch.empty(n, dtype=torch.long)
    for i in range(n):
        wa = bank["waves"][first[i], :bank["lens"][first[i]]].to(torch.float32) / 32767.0
        wb = bank["waves"][second[i], :bank["lens"][second[i]]].to(torch.float32) / 32767.0
        waves[i, 4352:4352 + len(wa)] = wa
        start2 = 4352 + len(wa) + 1600
        waves[i, start2:start2 + len(wb)] = wb
        lens[i] = start2 + len(wb)
    ids = 10**8 + pr.PAIRS.index(pair) * 10**7 + set_code * 10**5 + start + torch.arange(n)
    return pr.add_noise(waves, lens, ids, noise_db), lens


def test_the_order_task_builds_paper_02s_clips_exactly(bank):  # noqa: F811
    train_pool, _ = pr.protocol_a(bank)
    first, second, labels = pr.order_set(bank, train_pool, (3, 7), 12, 1)
    assert labels.tolist() == [0, 1] * 6
    mine = pr.order_clips(bank, first, second, (3, 7), 1, 0, 0.0)
    theirs = _paper02_order_clips(bank, first, second, (3, 7), 1, 0, 0.0)
    assert torch.equal(mine[0], theirs[0]) and torch.equal(mine[1], theirs[1])
    assert pr.joined_frames(2) == 147


def test_a_joined_clip_has_its_recordings_behind_the_leader_a_gap_apart(bank):  # noqa: F811
    train_pool, _ = pr.protocol_a(bank)
    idx, digits = pr.sequence_set(bank, train_pool, 3, 4, 1)
    waves, lens = pr.sequence_clips(bank, idx, 3, 1, 0, None)
    assert waves.shape == (4, pr.joined_samples(3)) == (4, 55552)
    for i in range(4):
        at = pr.LEADER_SAMPLES
        for p in range(3):
            n = int(bank["lens"][idx[i, p]])
            clip = bank["waves"][idx[i, p], :n].to(torch.float32) / pr.INT16_SCALE
            assert torch.equal(waves[i, at:at + n], clip)
            at += n + (pr.GAP_SAMPLES if p < 2 else 0)
        assert lens[i] == at
        assert waves[i, :pr.LEADER_SAMPLES].abs().sum() == 0
    assert [pr.joined_frames(n) for n in pr.SEQUENCE_LENGTHS] == [147, 216, 284]


@pytest.mark.parametrize("length", pr.SEQUENCE_LENGTHS)
def test_a_sequence_holds_different_digits_drawn_from_its_own_speakers(bank, length):  # noqa: F811
    train_pool, test_pool = pr.protocol_a(bank)
    for pool, code in ((train_pool, 2), (test_pool, 0)):
        idx, digits = pr.sequence_set(bank, pool, length, 500, code)
        assert all(len(set(row.tolist())) == length for row in digits)
        assert torch.equal(bank["labels"][idx], digits)
        assert torch.isin(idx, pool).all()
        counts = torch.bincount(digits[:, 0], minlength=10).double() / 500
        assert (counts - 0.1).abs().max() < 0.06, "every position is near uniform over the ten digits"
    again, _ = pr.sequence_set(bank, train_pool, length, 500, 2)
    assert torch.equal(again, pr.sequence_set(bank, train_pool, length, 500, 2)[0])


def test_the_chance_levels_are_exact():
    for length in pr.SEQUENCE_LENGTHS:
        c = pr.sequence_chance(length)
        assert c["per_position"] == 0.1 and c["order_free"] == pytest.approx(1 / length)
        assert c["whole"] == pytest.approx(1 / math.perm(10, length))
        assert c["whole_order_free"] == pytest.approx(1 / math.factorial(length))
    # with repeats: two digits are the same one time in ten, when a reader knowing the digits is right
    assert pr.order_free_ceiling(2, repeats=True) == pytest.approx(0.55)
    assert pr.order_free_ceiling(3, repeats=True) == pytest.approx(0.43)
    brute = sum(max(s.count(d) for d in s) for s in itertools.product(range(10), repeat=4)) / (4 * 10**4)
    assert pr.order_free_ceiling(4, repeats=True) == pytest.approx(brute) == pytest.approx(0.3835)


def test_the_order_task_runs_on_balanced_sets_and_reads_only_the_whole_span(bank, small_sets):  # noqa: F811
    s = spec(am.Arm("network", channels=2, grid=8), task="order", pair=(3, 7), sizes=(96,), native_sizes=(),
             widths=(64,), reads=("pooled",))
    assert s.group() == "size-order-spectrogram-8x8" and "/pair37/" in s.run_id()
    rec = rn.execute(s, bank=bank)
    assert {c["read"] for c in rec["cells"]} == {"pooled"} and rec["n_test"] == 64
    assert {c["projection"] for c in rec["cells"]} == {"fixed", "seeded"}


def test_the_sequence_task_reads_one_readout_per_position(bank, small_sets):  # noqa: F811
    s = spec(am.Arm("network", channels=2, grid=8), task="sequence", length=3, sizes=(120,), native_sizes=(),
             widths=(64,), reads=("pooled",))
    assert s.group() == "size-sequence-spectrogram-8x8" and "/seq3/" in s.run_id()
    rec = rn.execute(s, bank=bank)
    assert rec["n_test"] == 80
    assert {(c["position"], c["projection"]) for c in rec["cells"]} == {
        (p, proj) for p in range(3) for proj in ("fixed", "seeded")}
    assert all(0.0 <= c["acc"] <= 1.0 and "correct" in c for c in rec["cells"])


def test_the_spectrogram_only_baseline_reads_the_sequence_task_too(bank, small_sets):  # noqa: F811
    s = spec(am.Arm("baseline", grid=8), task="sequence", length=2, sizes=(120,), native_sizes=(120,),
             widths=(64,), reads=("pooled", "pooled@wholeclip"))
    rec = rn.execute(s, bank=bank)
    assert {c["position"] for c in rec["cells"]} == {0, 1}


def test_a_trained_baseline_is_not_run_on_the_sequence_task(bank, small_sets):  # noqa: F811
    with pytest.raises(ValueError, match="not run on the digit-sequence task"):
        rn.execute(spec(am.Arm("trained", arch="gru", channels=1, grid=8), task="sequence", length=2,
                        sizes=(120,)), bank=bank)

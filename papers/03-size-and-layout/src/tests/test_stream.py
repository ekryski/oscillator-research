"""The streamed read computes paper 02's read, channel by channel.

A large arm is simulated one channel at a time, standardized with its own
channel's training statistics, and projected into the run's features as it
goes (harness.confirm.stream). These tests hold it to the in-memory read:
given paper 02's projection matrix, rows matched to the channel's features,
it gives the same projected features to float32 rounding and the same
accuracy in every cell. With its own per-channel draw it is another fixed
Gaussian matrix of the same scale.
"""

import pytest
import torch

from harness.confirm import arms as am
from harness.confirm import readout as ro
from harness.confirm import run as rn
from harness.confirm import stream as st
from harness.measurement.features import projection_matrix

from .test_confirm_run import bank, spec  # noqa: F401  (the synthetic bank fixture)


def _global_rows(arm: am.Arm, c: int, windows: int = 4) -> torch.Tensor:
    """Where channel c's features sit in the in-memory read's feature order.

    In memory a read is ordered by window, statistic, then signal; the signals
    are sin theta for every oscillator of every channel, then cos theta (or
    one state per unit for the bank). A channel alone has the same order over
    its own signals.
    """
    sites = arm.grid * arm.grid
    halves = 2 if arm.kind == "field" else 1
    per_signal_block = halves * arm.channels * sites                 # D
    idx = []
    for j in range(windows):
        for s in range(3):
            for half in range(halves):
                start = j * 3 * per_signal_block + s * per_signal_block + half * arm.channels * sites + c * sites
                idx.append(torch.arange(start, start + sites))
    return torch.cat(idx)


def paper02_rows(arm: am.Arm):
    """Channel c's rows of the in-memory read's fixed (seed None) or seeded matrix."""
    def rows(native: int, c: int, n_rows: int, width: int, seed: int | None = None) -> torch.Tensor:
        idx = _global_rows(arm, c)
        assert len(idx) == n_rows
        return projection_matrix(native, width, seed)[idx]
    return rows


ARMS = [am.Arm("field", channels=4, grid=8), am.Arm("field", channels=3, grid=8, physics="winfree", boundary="cube"),
        am.Arm("field", channels=2, grid=8, severed=True), am.Arm("bank", channels=4, grid=8)]


@pytest.mark.parametrize("arm", ARMS, ids=lambda a: a.label())
def test_a_channel_alone_runs_exactly_as_it_does_among_the_others(arm):
    model = am.build_frozen(arm, 2.0, 1)
    rows = torch.rand(3, 30, arm.grid, generator=torch.Generator().manual_seed(2)) * 1.5
    with torch.no_grad():
        full = am.frozen_signals(arm, model, rows)
        sites = arm.grid * arm.grid
        for c in range(arm.channels):
            one, sub = am.channel(arm, model, c)
            alone = am.frozen_signals(one, sub, rows)
            if arm.kind == "field":
                cs = arm.channels * sites
                mine = torch.cat((full[..., c * sites:(c + 1) * sites], full[..., cs + c * sites:cs + (c + 1) * sites]), -1)
            else:
                mine = full[..., c * sites:(c + 1) * sites]
            assert torch.allclose(alone, mine, atol=1e-6), (arm.label(), c, (alone - mine).abs().max())


@pytest.mark.parametrize("arm", ARMS, ids=lambda a: a.label())
def test_with_paper_02s_matrix_the_streamed_read_is_the_in_memory_read(bank, arm, monkeypatch):  # noqa: F811
    s = spec(arm, sizes=(128,), widths=(64, 256, 1024), native_sizes=(), reads=("windowed",))
    monkeypatch.setattr(st, "STREAM_STATES", 10**9)
    held = rn.execute(s, bank=bank)
    monkeypatch.setattr(st, "STREAM_STATES", 0)
    monkeypatch.setattr(st, "channel_projection", paper02_rows(arm))
    streamed = rn.execute(s, bank=bank)
    assert held["read"] == "in memory" and streamed["read"] == "streamed by channel"
    assert held["native_widths"] == streamed["native_widths"]
    key = lambda c: (c["read"], c["width"], c["projection"])  # noqa: E731
    mine = {key(c): c for c in streamed["cells"]}
    assert {c["projection"] for c in held["cells"]} == {"fixed", "seeded"} and len(mine) == len(held["cells"])
    for c in held["cells"]:
        assert mine[key(c)]["acc"] == c["acc"] and mine[key(c)]["lam"] == c["lam"], key(c)
        assert mine[key(c)]["correct"] == c["correct"]


def test_the_projected_features_agree_to_float32_rounding(bank, monkeypatch):  # noqa: F811
    arm = am.Arm("field", channels=4, grid=8)
    s = spec(arm, sizes=(128,), widths=(64, 256), native_sizes=(), reads=("windowed",))
    clips = rn.assemble(s, bank)
    model = am.build_frozen(arm, 2.0, 0)
    held = []
    with torch.no_grad():
        for rows, tvalid, _ in rn.batches(s, clips):
            held.append(am.frozen_features(arm, am.frozen_signals(arm, model, rows), tvalid, "recognition")["windowed"])
    held = torch.cat(held)
    n, test = 128, clips.layout.test
    mean, sd = ro._stats(held[:n])
    with torch.no_grad():
        streamed, native = st.streamed_read(arm, model, rn.channel_batches(s, clips, n), n, clips.layout.n_test,
                                            (64, 256), "recognition", 2, projection=paper02_rows(arm))
    assert native == held.shape[1]
    for projection, seed in (("fixed", None), ("seeded", 2)):
        p = projection_matrix(native, 256, seed)
        want_tr = ro._projected([held], slice(0, n), mean, sd, 256, p)
        want_te = ro._projected([held], test, mean, sd, 256, p)
        p_tr, p_te = streamed[projection]
        assert torch.allclose(p_tr, want_tr, rtol=1e-4, atol=1e-5)
        assert torch.allclose(p_te, want_te, rtol=1e-4, atol=1e-5)


def test_the_streamed_matrices_are_fixed_or_seeded_gaussians_scaled_like_paper_02s():
    a, b = st.channel_projection(10_000, 3, 500, 256), st.channel_projection(10_000, 3, 500, 256)
    assert torch.equal(a, b) and a.shape == (500, 256)
    assert not torch.equal(a, st.channel_projection(10_000, 4, 500, 256))
    assert (st.channel_seed(3), st.channel_seed(3, 0), st.channel_seed(5, 2)) == (424_203, 424_303, 424_505)
    seeded = [st.channel_projection(10_000, 3, 500, 256, s) for s in (0, 1)]
    assert not torch.equal(seeded[0], seeded[1]) and not torch.equal(seeded[0], a)
    assert torch.equal(st.channel_projection(10_000, 3, 500, 64), a[:, :64])       # narrow = leading columns
    assert abs(a.std().item() * 10_000 ** 0.5 - 1.0) < 0.02


def test_a_large_arm_is_streamed_and_one_paper_02_read_in_memory():
    assert rn.Spec("size", "recognition", "envelope", 0.0, 1.0, 0, am.Arm("field", channels=16, grid=32)).streamed
    assert not rn.Spec("size", "recognition", "envelope", 0.0, 1.0, 0, am.Arm("field", channels=16)).streamed
    assert not rn.Spec("size", "recognition", "envelope", 0.0, 1.0, 0, am.Arm("field", grid=64, channels=1)).streamed
    assert rn.Spec("size", "recognition", "envelope", 0.0, 1.0, 0, am.Arm("bank", grid=128, channels=1)).streamed


def test_with_no_input_the_streamed_read_is_chance(bank, monkeypatch):  # noqa: F811
    monkeypatch.setattr(st, "STREAM_STATES", 0)
    rec = rn.execute(spec(am.Arm("field", channels=2, grid=8), gain=0.0, sizes=(128,), widths=(64, 256),
                          native_sizes=(), reads=("windowed",)), bank=bank)
    assert rec["read"] == "streamed by channel"
    assert {c["acc"] for c in rec["cells"]} == {0.1}

"""The shared readout: the closed-form ridge at any size, and every arm read the same way."""

import pytest
import torch

from harness.experiment import readout as ro
from harness.measurement.features import projection_matrix


def blobs(n, d, classes=4, seed=0, spread=1.5):
    """Noisy class clusters: separable enough to learn, hard enough that the penalty matters."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(classes, d, generator=g) * spread / d ** 0.5
    y = torch.arange(n) % classes
    return centres[y] + torch.randn(n, d, generator=g) / d ** 0.5 * 3, y


def test_primal_and_dual_are_the_same_estimator():
    x, y = blobs(500, 50, seed=3)
    z = ro._design(x, *ro._stats(x))
    y1h = torch.nn.functional.one_hot(y, 4).double()
    a, b = ro._gram(x, *ro._stats(x), y1h, slice(0, 500))
    primal = ro._solve_primal(a, b, 0.01, 500)
    dual = ro._solve_dual(z, z @ z.T, y1h, 0.01)
    assert torch.allclose(primal, dual, atol=1e-8)


def test_an_explicit_validation_set_chooses_the_penalty_and_training_is_not_split():
    x, y = blobs(600, 30, seed=5)
    r = ro.ridge(x[:300], y[:300], x[450:], y[450:], 4, x_val=x[300:450], y_val=y[300:450])
    fitted_on_all = ro.ridge(x[:300], y[:300], x[450:], y[450:], 4, x_val=x[300:450], y_val=y[300:450],
                             lambdas=(r["lam"],))
    assert r["correct"].tolist() == fitted_on_all["correct"].tolist()


def test_per_clip_correctness_survives_packing():
    c = torch.rand(6000, generator=torch.Generator().manual_seed(1)) > 0.3
    assert torch.equal(ro.unpack(ro.pack(c), 6000), c)
    assert len(ro.pack(c)) <= 1000


def test_projection_is_of_the_standardized_features():
    x = torch.randn(300, 90, generator=torch.Generator().manual_seed(2)) * 5 + 3
    mean, sd = ro._stats(x[:200])
    got = ro._projected([x], slice(0, 300), mean, sd, 16)
    explicit = (((x.double() - mean) / sd) @ projection_matrix(90, 16).double()).float()
    assert torch.allclose(got, explicit, atol=1e-4)


def test_a_feature_constant_over_training_is_given_no_weight():
    x = torch.randn(300, 90, generator=torch.Generator().manual_seed(6))
    x[:, 7] = 5.0
    x[250:, 7] = 5.0 + 1e-3            # constant in training, jittering only outside it
    mean, sd = ro._stats(x[:200])
    got = ro._projected([x], slice(0, 300), mean, sd, 16)
    x0 = x.clone(); x0[:, 7] = 0.0
    assert torch.allclose(got, ro._projected([x0], slice(0, 300), *ro._stats(x0[:200]), 16), atol=1e-5)


def test_a_read_in_blocks_is_read_as_its_concatenation():
    x = torch.randn(300, 90, generator=torch.Generator().manual_seed(4))
    mean, sd = ro._stats(x[:200])
    whole = ro._projected([x], slice(0, 300), mean, sd, 16)
    split = ro._projected([x[:, :60], x[:, 60:]], slice(0, 300), mean, sd, 16)
    assert torch.allclose(whole, split, atol=1e-5)


def cells_for(native=40, n_train=320, n_test=160, sizes=(160, 320), widths=(16, 64), native_sizes=(160,)):
    x, y = blobs(n_train + n_test, native, seed=7)
    layout = ro.Layout(n_train, 0, n_test)
    return ro.read_cells({"windowed": [x]}, y, layout, sizes, widths, native_sizes, 4,
                         keep_bits=lambda read, width, n: width == 16), x, y, layout


def test_every_size_and_width_gets_a_cell_and_wide_widths_read_the_native_features():
    cells, *_ = cells_for()
    got = {(c["n_train"], c["width"]): c for c in cells}
    assert set(got) == {(160, 16), (160, 64), (160, "native"), (320, 16), (320, 64)}
    # 64 is wider than the 40 native features, so it is the native read, fitted once
    assert got[(160, 64)]["effective_width"] == 40 and got[(160, 64)]["acc"] == got[(160, "native")]["acc"]
    assert got[(160, 16)]["effective_width"] == 16


def test_a_training_size_is_the_first_n_rows_and_the_test_rows_never_move():
    cells, x, y, layout = cells_for()
    manual = ro.ridge(x[:160], y[:160], x[layout.test], y[layout.test], 4)
    got = next(c for c in cells if (c["n_train"], c["width"]) == (160, "native"))
    assert got["acc"] == manual["acc"] and got["lam"] == manual["lam"]


def test_only_the_cells_asked_for_keep_their_per_clip_bits():
    cells, *_ = cells_for()
    assert all(("correct" in c) == (c["width"] == 16) for c in cells)


def test_a_size_larger_than_the_training_rows_is_an_error():
    with pytest.raises(ValueError, match="exceeds"):
        cells_for(sizes=(400,))


def test_the_pca_projection_keeps_the_training_rows_leading_variance_and_never_sees_the_test_rows():
    from harness.experiment.readout import _block_stats, pca_matrix
    gen = torch.Generator().manual_seed(0)
    low = torch.randn(300, 5, generator=gen) @ torch.randn(5, 400, generator=gen)   # rank-5 structure
    x = low + 0.01 * torch.randn(300, 400, generator=gen)
    blocks = [x]
    mean, sd = _block_stats(blocks, slice(0, 200))
    p = pca_matrix(blocks, slice(0, 200), mean, sd, 5)
    assert p.shape == (400, 5)
    z = ((x[:200] - mean.float()) / sd.float())
    kept = (z @ p).pow(2).sum() / z.pow(2).sum()
    assert kept > 0.99, kept                                   # five directions carry the rank-5 structure
    other = pca_matrix([torch.cat((x[:200], x[200:] * 100))], slice(0, 200), mean, sd, 5)
    assert torch.allclose(p, other)                            # rows beyond the training slice change nothing


def test_the_pooled_projection_averages_each_statistic_over_a_lattice_rows_signals():
    from harness.experiment.arms import Arm, signal_rows
    from harness.experiment.readout import pooling_matrix
    rows = signal_rows(Arm("network", channels=2), grid=4)         # sin then cos of 2 x 4 x 4 oscillators
    assert rows.shape == (64,) and rows.tolist()[:8] == [0, 0, 0, 0, 1, 1, 1, 1]
    p = pooling_matrix(rows, stats=3, windows=2)
    assert p.shape == (2 * 3 * 64, 2 * 3 * 4)
    assert torch.allclose(p.sum(0), torch.ones(24))                 # every pooled feature is a mean
    feats = torch.arange(2 * 3 * 64, dtype=torch.float32)[None]     # feature f has value f
    pooled = feats @ p                                              # [1, 24]
    # window 1, statistic 2, row 3 averages the features of that window and statistic whose signal sits in row 3
    f0 = (1 * 3 + 2) * 64
    expect = torch.tensor([f0 + d for d in range(64) if rows[d] == 3], dtype=torch.float32).mean()
    assert pooled[0, (1 * 3 + 2) * 4 + 3] == expect
    assert signal_rows(Arm("bank", channels=2), grid=4).shape == (32,)
    assert signal_rows(Arm("baseline"), grid=4).tolist() == [0, 1, 2, 3]     # the baseline's signals are its bands
    assert signal_rows(Arm("trained", arch="gru")) is None


def test_a_row_restricted_pooling_keeps_one_half_of_the_rows_in_the_same_order():
    from harness.experiment.arms import Arm, signal_rows
    from harness.experiment.readout import pooling_matrix
    rows = signal_rows(Arm("network", channels=2), grid=4)
    full = pooling_matrix(rows, stats=3, windows=2)
    lower = pooling_matrix(rows, stats=3, windows=2, keep_rows=(0, 1))
    upper = pooling_matrix(rows, stats=3, windows=2, keep_rows=(2, 3))
    assert lower.shape == upper.shape == (2 * 3 * 64, 2 * 3 * 2)
    keep_lower = [c for c in range(24) if c % 4 in (0, 1)]
    assert torch.equal(lower, full[:, keep_lower])
    assert torch.equal(upper, full[:, [c for c in range(24) if c % 4 in (2, 3)]])
    # the baseline's pooling is the identity, so a half keeps that half of its bands
    bands = signal_rows(Arm("baseline"), grid=4)
    assert torch.equal(pooling_matrix(bands, 3, 1), torch.eye(12))
    assert torch.equal(pooling_matrix(bands, 3, 1, keep_rows=(2, 3)), torch.eye(12)[:, [2, 3, 6, 7, 10, 11]])
    from harness.experiment.readout import _kept_rows
    assert _kept_rows("pooled-row3", rows) == (3,) and _kept_rows("pooled-lower", rows) == (0, 1)
    assert _kept_rows("pooled", rows) is None


def test_read_cells_reads_through_pca_and_pooling_at_the_pooled_width():
    from harness.experiment.arms import Arm, signal_rows
    gen = torch.Generator().manual_seed(1)
    n, d, grid = 120, 2 * 2 * 4 * 4, 4                         # a 2-channel 4 x 4 network's 64 signals
    windows = 2
    feats = torch.randn(n, windows * 3 * d, generator=gen)
    labels = torch.randint(0, 3, (n,), generator=gen)
    layout = ro.Layout(80, 0, 40)
    keep = lambda read, width, size: False  # noqa: E731
    rows = signal_rows(Arm("network", channels=2), grid=grid)
    for method in ("pca", "pooled"):
        cells = ro.read_cells({"windowed": [feats]}, labels, layout, (80,), (windows * 3 * grid,), (), 3, keep,
                              method=method, signal_rows=rows, windows=windows)
        assert len(cells) == 1 and cells[0]["effective_width"] == windows * 3 * grid
        assert 0.0 <= cells[0]["acc"] <= 1.0
    for method in ("pooled-lower", "pooled-upper"):
        cells = ro.read_cells({"windowed": [feats]}, labels, layout, (80,), (windows * 3 * grid // 2,), (), 3, keep,
                              method=method, signal_rows=rows, windows=windows)
        assert len(cells) == 1 and cells[0]["effective_width"] == windows * 3 * grid // 2
        assert 0.0 <= cells[0]["acc"] <= 1.0
    cells = ro.read_cells({"windowed": [feats]}, labels, layout, (80,), (windows * 3,), (), 3, keep,
                          method="pooled-row2", signal_rows=rows, windows=windows)
    assert len(cells) == 1 and cells[0]["effective_width"] == windows * 3 and 0.0 <= cells[0]["acc"] <= 1.0
    with pytest.raises(ValueError):
        ro.read_cells({"windowed": [feats]}, labels, layout, (80,), (30,), (), 3, keep, method="pooled",
                      signal_rows=rows, windows=windows)

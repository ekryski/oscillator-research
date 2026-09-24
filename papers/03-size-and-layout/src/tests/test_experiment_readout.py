"""The shared readout: paper 02's ridge at any size, and every arm read the same way."""

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
    x0 = x.clone()
    x0[:, 7] = 0.0
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

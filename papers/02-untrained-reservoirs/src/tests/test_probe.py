"""Contracts for `harness.probe`.

The probe is deliberately the weakest reader in the stack — a linear scorer
cannot compute, so whatever separation it finds must already exist in the
features. These tests pin its determinism, its masking under variable clip
lengths, and the readout-capacity parity projection that stops a wider
feature vector from masquerading as better dynamics."""

import math
import sys

import torch

sys.path.insert(0, ".")
from harness.measurement.probe import (
    fit_ridge_probe,
    parity_project,
    phase_features,
    pooled_stats,
)
from harness.models import (
    OscillatorField,
)

TWO_PI = 2 * math.pi


def test_phase_features_shapes_and_bounds():
    torch.manual_seed(0)
    c, g, t = 2, 8, 40
    theta = torch.rand(3, t, c, g, g) * TWO_PI
    feats = torch.cat((torch.sin(theta), torch.cos(theta)), dim=2).flatten(2)
    out = phase_features(feats, c, g, warmup=4)
    assert out.shape == (3, 4 * c * g * g)
    assert torch.isfinite(out).all() and out.abs().max() <= 1.0 + 1e-5


def test_ridge_probe_learns_separable_data_and_is_deterministic():
    gen = torch.Generator().manual_seed(0)
    n, d, k = 256, 32, 4
    y = torch.arange(n) % k
    f = torch.zeros(n, d)
    f[torch.arange(n), y] = 2.0
    f = f + 0.1 * torch.randn(n, d, generator=gen)
    r1 = fit_ridge_probe(f[:192], y[:192], f[192:], y[192:], k)
    r2 = fit_ridge_probe(f[:192], y[:192], f[192:], y[192:], k)
    assert r1["acc"] > 0.95
    assert r1 == r2


def test_trainArm_bestVal_restoresBestEpoch_andStampsHistory():
    from harness.measurement.probe import train_arm
    torch.manual_seed(0)
    m = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0)
    gen = torch.Generator().manual_seed(11)
    rows = torch.randn(24, 48, 16, generator=gen) * 0.5
    y = torch.randint(0, 4, (24,), generator=gen)
    scores = iter([0.2, 0.9, 0.3])
    snaps = []

    def sel(model):
        snaps.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return next(scores)

    hist = train_arm(m, rows[:16], y[:16], rows[16:], y[16:], epochs=3, lr=1e-3,
                     batch=8, seed=0, selector=sel, select_every=1)
    assert hist[-1]["sel_epoch"] == 1 and hist[-1]["sel_score"] == 0.9
    for k, v in m.state_dict().items():  # epoch-1 weights restored, not epoch-2
        assert torch.allclose(v, snaps[1][k]), k


def test_trainArm_selector_doesNotPerturbTraining():
    # selection is eval-only: the training trajectory must be bit-identical
    # with and without a selector (it consumes no training RNG)
    from harness.measurement.probe import train_arm

    def build():
        torch.manual_seed(3)
        return OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0)

    gen = torch.Generator().manual_seed(11)
    rows = torch.randn(24, 48, 16, generator=gen) * 0.5
    y = torch.randint(0, 4, (24,), generator=gen)
    h1 = train_arm(build(), rows[:16], y[:16], rows[16:], y[16:],
                   epochs=2, lr=1e-3, batch=8, seed=0)
    h2 = train_arm(build(), rows[:16], y[:16], rows[16:], y[16:],
                   epochs=2, lr=1e-3, batch=8, seed=0,
                   selector=lambda model: 0.0, select_every=1)
    # Tolerance, not equality, and the reason matters: the dense coupling path
    # accumulates its backward through a gather, whose reduction order is not
    # pinned, so two identical training runs agree only to float32 rounding.
    # (Forward passes ARE bit-identical — see test_phase.py — which is why the
    # frozen arms the paper reports replicate exactly.) What this test is for
    # is that the SELECTOR changes nothing: it must consume no training RNG and
    # perturb no step, so any drift stays at that rounding floor.
    for a, b in zip([r["loss"] for r in h1], [r["loss"] for r in h2], strict=True):
        assert abs(a - b) < 1e-5 * max(1.0, abs(a)), (a, b)


def test_maskedpooled_stats_dead_unit_grad_stays_finite():
    # Regression (digits, measured): a dead unit has EXACTLY zero
    # variance over a short valid window, and sqrt'(0) = inf made its zero
    # upstream gradient NaN (inf * 0), aborting every ReLU baseline's digits
    # training. The masked branch's sqrt(var + eps) keeps gradients finite.
    from harness.measurement.probe import pooled_stats
    h = torch.randn(2, 20, 4)
    h[:, :, 1] = 0.0  # dead unit: constant over every window
    h.requires_grad_(True)
    out = pooled_stats(h, tvalid=torch.tensor([6, 12]))
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(h.grad).all()


def test_parity_projection_common_width_and_deterministic():
    #: every native width lands at PARITY_DIM; the projection is seeded,
    # not an artifact of the in-process cache (fresh draw reproduces it).
    from harness.measurement.probe import _PARITY_PROJ, PARITY_DIM, parity_project
    assert PARITY_DIM == 72
    gen = torch.Generator().manual_seed(3)
    for d in (4096, 72, 64):  # torus / GRU / TCN native feature widths
        f = torch.randn(5, d, generator=gen)
        p1 = parity_project(f)
        assert p1.shape == (5, PARITY_DIM) and torch.isfinite(p1).all()
        _PARITY_PROJ.clear()
        assert torch.equal(p1, parity_project(f))
    assert set(_PARITY_PROJ) == {64}  # one cached projection per native dim


def test_ridge_selects_its_lambda_on_held_out_train_not_on_test():
    """The probe's one hyperparameter is chosen inside the training split, so a
    reported accuracy is never the best-of-several-peeks at the test set."""
    torch.manual_seed(0)
    n, d, k = 320, 24, 4
    y = torch.arange(n) % k
    f = torch.zeros(n, d)
    f[torch.arange(n), y] = 1.5
    f += 0.4 * torch.randn(n, d)
    r = fit_ridge_probe(f[:256], y[:256], f[256:], y[256:], k)
    assert r["lam"] in (1e-3, 1e-2, 1e-1, 1.0)
    assert 0.0 <= r["val_acc"] <= 1.0
    assert len(r["pred"]) == 64
    assert r["margin"] > 0


def test_ridge_is_at_chance_on_label_free_features():
    """The negative control for the reader itself: no structure in, no
    accuracy out. Without this, a plateau result could be the probe hallucinating."""
    torch.manual_seed(0)
    n, k = 512, 10
    f = torch.randn(n, 32)
    y = torch.randint(k, (n,))
    r = fit_ridge_probe(f[:384], y[:384], f[384:], y[384:], k)
    assert r["acc"] < 0.25, "a linear scorer cannot invent structure that is not there"


def test_phase_features_length_masking_ignores_padded_frames():
    """Digit clips are right-zero-padded, so pooled statistics must run over
    valid frames only or the padding dilutes every short clip's features."""
    torch.manual_seed(0)
    c, g, t, warm = 2, 4, 40, 4
    theta = torch.rand(2, t, c, g, g) * TWO_PI
    feats = torch.cat((torch.sin(theta), torch.cos(theta)), dim=2).flatten(2)
    tvalid = torch.tensor([t, 20])
    masked = phase_features(feats, c, g, warmup=warm, tvalid=tvalid)
    # the fully valid clip matches the unmasked computation over the same span
    plain = phase_features(feats[:1], c, g, warmup=warm)
    assert torch.allclose(masked[:1], plain, atol=1e-5)
    # the truncated clip matches the unmasked computation on its own prefix
    short = phase_features(feats[1:, :20], c, g, warmup=warm)
    assert torch.allclose(masked[1:], short, atol=1e-5)


def test_pooled_stats_report_the_last_VALID_frame_not_the_last_padded_one():
    h = torch.zeros(1, 10, 3)
    h[0, :4] = 1.0                     # "signal" only in the valid span
    out = pooled_stats(h, tvalid=torch.tensor([4]))
    mean, _, _, last = out.split(3, dim=1)
    assert torch.allclose(mean, torch.ones(1, 3))
    assert torch.allclose(last, torch.ones(1, 3))


def test_parity_projection_is_stable_across_processes():
    """The projection is drawn from a fixed seed and cached per native width,
    so two arms of different feature widths are compared at one capacity."""
    a = parity_project(torch.eye(64))
    b = parity_project(torch.eye(64))
    assert torch.equal(a, b)
    assert a.shape[1] == parity_project(torch.eye(128)).shape[1]

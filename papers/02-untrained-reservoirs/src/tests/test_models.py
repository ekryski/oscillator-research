"""Contracts for `harness.models`.

The physics-only invariant is the load-bearing one: gradients may touch the
coupling kernel and the natural frequencies and nothing else, so that any
learned gain is attributable to the dynamics rather than to scaffolding.
The rest pin the feature contracts every arm shares."""

import math
import sys

import torch

sys.path.insert(0, ".")
from harness.models import (
    GRUBaseline,
    OscillatorField,
)
from harness.models.phase import PhaseBlock
from harness.stimuli import (
    band_edges,
    band_index,
    bandpass_rows,
)

TWO_PI = 2 * math.pi


def test_physics_only_training_updates_only_physics():
    torch.manual_seed(0)
    model = OscillatorField(channels=2, grid=8, n_classes=4, probe_seed=1)
    trainable = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    assert trainable == ["core.blocks.0.kernel", "core.blocks.0.natural_freqs"]
    probe_before = model.probe_w.clone()
    k_before = model.core.blocks[0].kernel.detach().clone()
    rows = torch.randn(4, 64, 8) * 0.5
    labels = torch.tensor([0, 1, 2, 3])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    loss = torch.nn.functional.cross_entropy(model(rows), labels)
    loss.backward()
    opt.step()
    assert torch.equal(model.probe_w, probe_before)
    assert not torch.equal(model.core.blocks[0].kernel.detach(), k_before)


def test_gru_param_count_matches_physics_budget():
    torch.manual_seed(0)
    gru = GRUBaseline(grid=16, hidden=18, n_classes=8)
    torus = OscillatorField(channels=4, grid=16, n_classes=8)
    n_gru = sum(p.numel() for p in gru.parameters() if p.requires_grad)
    n_phys = sum(p.numel() for p in torus.parameters() if p.requires_grad)
    assert n_phys == 2 * 4 * 16 * 16
    assert abs(n_gru - n_phys) / n_phys < 0.15


def test_drive_response_localizes_to_driven_row_and_drags_phase():
    # Drive enters ADDITIVELY on phase velocity, so a cold field cannot 1:1
    # phase-lock to a carrier far from its natural rate — locking exists only
    # through the damping washboard's narrow Shapiro/Adler tongues (measured
    # measured: gain 2 drags phase, gain 6 lands near the J1 Bessel null and
    # scrambles it; see the training-log design note). The hard contracts:
    # (a) the velocity-signature response localizes to the driven row, and
    # (b) at the big-model default gain 2.0 the driven row's PLV vs the
    #     stimulus exceeds the free-running baseline (partial phase dragging).
    torch.manual_seed(0)
    grid, t = 16, 512
    e = band_edges(grid)
    f = float(math.sqrt(e[10] * e[11]))  # ~0.037 cyc/frame vs natural ~0.014
    row = band_index(f, grid)
    phase = TWO_PI * f * torch.arange(t)
    rows = bandpass_rows(torch.sin(phase)[None, :], grid)
    warm = 16

    def field(model, rows_in):
        theta = model.phase_trajectory(rows_in)[:, warm:]  # [1,T,C,G,G]
        diff = theta - phase[None, warm:, None, None, None]
        plv = torch.complex(torch.cos(diff).mean(dim=1), torch.sin(diff).mean(dim=1)).abs()[0]
        vel = torch.cos(theta[:, 1:] - theta[:, :-1]).mean(dim=1)[0]
        return plv, vel

    strong = OscillatorField(channels=2, grid=grid, n_classes=2, probe_seed=0, gain=6.0)
    _, vel_d = field(strong, rows)
    _, vel_f = field(strong, torch.zeros_like(rows))
    dvel = (vel_d - vel_f).abs()
    far = (row + 8) % grid
    assert dvel[:, row].mean() > 0.03  # measured 0.087
    assert dvel[:, row].mean() > 20 * dvel[:, far].mean()  # measured ratio ~200x

    default = OscillatorField(channels=2, grid=grid, n_classes=2, probe_seed=0, gain=2.0)
    plv_d, _ = field(default, rows)
    plv_f, _ = field(default, torch.zeros_like(rows))
    assert plv_d[:, row].mean() > plv_f[:, row].mean() + 0.05  # measured 0.146 vs 0.034


def test_coupling_variants_match_kuramoto_at_zero_init():
    # sakaguchi(alpha=0) and harmonic2(beta=0) are EXACTLY kuramoto — the
    # stage-D arms start warm-comparable by construction.
    rows = torch.randn(2, 48, 16) * 0.5
    feats = {}
    for coupling in ("kuramoto", "sakaguchi", "harmonic2"):
        torch.manual_seed(3)
        m = OscillatorField(channels=2, grid=16, coupling=coupling, n_classes=4, probe_seed=0)
        feats[coupling] = m.features(rows)
    assert torch.allclose(feats["kuramoto"], feats["sakaguchi"], atol=1e-6)
    assert torch.allclose(feats["kuramoto"], feats["harmonic2"], atol=1e-6)


def test_kernel_support_zeroes_far_taps():
    torch.manual_seed(0)
    blk = PhaseBlock(channels=1, grid=8, kernel_support=1, coupling_impl="fft")
    k_eff = torch.fft.irfft2(blk.prepare_coupling(), s=(8, 8))
    off = torch.arange(8)
    d = torch.minimum(off, 8 - off)
    far = (d[:, None] > 1) | (d[None, :] > 1)
    assert k_eff[0][far].abs().max() < 1e-6
    assert k_eff[0][~far].abs().max() > 1e-3  # near taps survive


def test_blocks2_keeps_physics_only_trainable():
    torch.manual_seed(0)
    m = OscillatorField(channels=2, grid=8, blocks=2, n_classes=4, probe_seed=0)
    trainable = sorted(n for n, p in m.named_parameters() if p.requires_grad)
    assert trainable == ["core.blocks.0.kernel", "core.blocks.0.natural_freqs",
                        "core.blocks.1.kernel", "core.blocks.1.natural_freqs"]
    assert all(not p.requires_grad for p in m.core.mixers.parameters())


def test_tonotopic_omega_rows_match_band_centers():
    from harness import tonotopic_omega
    w = tonotopic_omega(2, 16, 0.1, 1, torch.Generator().manual_seed(0))
    assert w.shape == (2, 16, 16)
    e = band_edges(16)
    for r in (3, 9, 15):
        want = TWO_PI * math.sqrt(float(e[r] * e[r + 1])) / 0.1
        assert abs(w[:, r].mean().item() - want) / want < 0.03


def test_learnable_damping_matches_fixed_at_init_and_is_physics():
    rows = torch.randn(2, 48, 16) * 0.5
    feats = {}
    for core in ("phase", "sl"):
        for learn in (False, True):
            torch.manual_seed(4)
            m = OscillatorField(channels=2, grid=16, damping=0.1, n_classes=4,
                         probe_seed=0, core=core, damping_learnable=learn)
            feats[(core, learn)] = m.features(rows)
            if learn:
                names = [n for n, p in m.named_parameters()
                         if p.requires_grad and n.endswith("damping_lam")]
                assert len(names) == 1  # per-channel lambda is trainable physics
    assert torch.allclose(feats[("phase", False)], feats[("phase", True)], atol=1e-6)
    assert torch.allclose(feats[("sl", False)], feats[("sl", True)], atol=1e-6)


def test_winfree_and_omegaenc_arms_contract():
    rows = torch.randn(2, 48, 16) * 0.5
    # winfree: runs finite, its S/I params are trainable physics
    torch.manual_seed(5)
    m = OscillatorField(channels=2, grid=16, coupling="winfree", n_classes=4, probe_seed=0)
    assert torch.isfinite(m.features(rows)).all()
    names = sorted(n for n, p in m.named_parameters() if p.requires_grad)
    assert any(n.endswith("winfree_s") for n in names)
    # omegaenc: zero-init encoder => identical to the undriven frozen field
    torch.manual_seed(5)
    enc = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0, omega_encoder=True)
    torch.manual_seed(5)
    plain = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0)
    free = plain.features(torch.zeros_like(rows))
    assert torch.allclose(enc.features(rows), free, atol=1e-6)
    # windowed features: 4x the standard dim, finite
    fw = plain.features_windowed(rows, windows=4)
    assert fw.shape[1] == 4 * plain.feat_dim and torch.isfinite(fw).all()


def test_tcn_reference_matches_budget_and_runs():
    from harness import TCNBaseline
    torch.manual_seed(0)
    m = TCNBaseline(grid=16, n_classes=8)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert abs(n - 2048) / 2048 < 0.15  # matched to the physics budget
    out = m(torch.randn(3, 64, 16))
    assert out.shape == (3, 8) and torch.isfinite(out).all()


def test_features_macro_contract():
    #: patch-R + row-|z| blocks; R in [0,1]; phase-core amp block == 1
    rows = torch.randn(3, 64, 16) * 0.5
    torch.manual_seed(2)
    phase = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0)
    fm = phase.features_macro(rows, patch=4)
    n_patch, n_row = 2 * 16, 2 * 16  # C*(G/4)^2, C*G
    assert fm.shape == (3, 2 * n_patch + 2 * n_row) and torch.isfinite(fm).all()
    r_mean = fm[:, :n_patch]
    assert (r_mean >= 0).all() and (r_mean <= 1 + 1e-5).all()
    amp_mean, amp_std = fm[:, 2 * n_patch:2 * n_patch + n_row], fm[:, 2 * n_patch + n_row:]
    assert torch.allclose(amp_mean, torch.ones_like(amp_mean), atol=1e-5)  # phase core: |z| == 1
    assert amp_std.abs().max() < 1e-5
    torch.manual_seed(2)
    sl = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0, core="sl")
    fs = sl.features_macro(rows, patch=4)
    assert fs.shape == fm.shape and torch.isfinite(fs).all()
    assert fs[:, 2 * n_patch:2 * n_patch + n_row].std() > 1e-4  # SL amp varies


def test_features_settle_contract():
    # settle-read: drive-free continuation features are finite, keep the
    # arm's native feat_dim, and genuinely differ from the standard features —
    # for the phase core, the SL core, and both conventional references.
    from harness import TCNBaseline
    torch.manual_seed(1)
    rows = torch.rand(3, 64, 16) * 0.5
    for core in ("phase", "sl"):
        torch.manual_seed(2)
        m = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0, core=core)
        fs = m.features_settle(rows)
        assert fs.shape == (3, m.feat_dim) and torch.isfinite(fs).all()
        assert not torch.allclose(fs, m.features(rows), atol=1e-3)
    for cls in (GRUBaseline, TCNBaseline):
        torch.manual_seed(0)
        ref = cls(grid=16, n_classes=4)
        fs = ref.features_settle(rows)
        assert fs.shape == (3, ref.feat_dim) and torch.isfinite(fs).all()
        assert not torch.allclose(fs, ref.features(rows), atol=1e-3)
    # the quad frontend settles too (digits phase factor rides the column)
    torch.manual_seed(2)
    m = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0)
    phi = torch.rand(3, 64, 16) * TWO_PI
    quad = torch.stack((rows * torch.cos(phi), rows * torch.sin(phi)), dim=-1)
    fq = m.features_settle(quad)
    assert fq.shape == (3, m.feat_dim) and torch.isfinite(fq).all()


def test_masked_windowed_and_macro_features():
    # length masking: tvalid == T reproduces the unmasked features
    # exactly; shorter tvalid changes them; short clips stay finite (window
    # rule borrows trailing frames). Same contract for the conventional refs.
    from harness import TCNBaseline
    torch.manual_seed(1)
    rows = torch.rand(3, 64, 16) * 0.5
    t = rows.shape[1]
    full = torch.full((3,), t, dtype=torch.long)
    short = torch.tensor([30, 45, t])
    tiny = torch.tensor([18, 18, 18])  # < WARMUP + 2*windows: borrow rule kicks in
    torch.manual_seed(2)
    m = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0)
    fw = m.features_windowed(rows, windows=4)
    assert torch.allclose(fw, m.features_windowed(rows, windows=4, tvalid=full), atol=1e-6)
    fw_short = m.features_windowed(rows, windows=4, tvalid=short)
    assert fw_short.shape == fw.shape and torch.isfinite(fw_short).all()
    assert not torch.allclose(fw_short[0], fw[0], atol=1e-4)  # masked clip differs
    assert torch.allclose(fw_short[2], fw[2], atol=1e-6)      # full-length clip unchanged
    assert torch.isfinite(m.features_windowed(rows, windows=4, tvalid=tiny)).all()
    fm = m.features_macro(rows, patch=4)
    assert torch.allclose(fm, m.features_macro(rows, patch=4, tvalid=full), atol=1e-5)
    fm_short = m.features_macro(rows, patch=4, tvalid=short)
    assert torch.isfinite(fm_short).all()
    assert not torch.allclose(fm_short[0], fm[0], atol=1e-4)
    assert torch.allclose(fm_short[2], fm[2], atol=1e-5)
    for cls in (GRUBaseline, TCNBaseline):
        torch.manual_seed(0)
        ref = cls(grid=16, n_classes=4)
        gw = ref.features_windowed(rows, windows=4)
        assert torch.allclose(gw, ref.features_windowed(rows, windows=4, tvalid=full), atol=1e-6)
        gw_short = ref.features_windowed(rows, windows=4, tvalid=short)
        assert torch.isfinite(gw_short).all()
        assert not torch.allclose(gw_short[0], gw[0], atol=1e-4)
        assert torch.allclose(gw_short[2], gw[2], atol=1e-6)
        assert torch.isfinite(ref.features_windowed(rows, windows=4, tvalid=tiny)).all()


def test_coupling_laws_distinct_at_nonzero_init():
    #: frozen sakaguchi/harmonic2 must be REAL factor levels when
    # given nonzero inits — and the zero-init degeneracy (== kuramoto) is
    # intentional training-start semantics, asserted here so it can never
    # silently re-enter an untrained matrix unnoticed.
    import math

    from harness import OscillatorField

    torch.manual_seed(3)
    rows = torch.rand(2, 40, 16) * 0.5
    def feats(coupling, **kw):
        torch.manual_seed(11)  # identical K/omega/theta0 draws across laws
        m = OscillatorField(channels=2, grid=16, n_classes=4, probe_seed=0,
                     coupling=coupling, **kw)
        return m.features(rows)
    kur = feats("kuramoto")
    # zero-init degeneracy: documented, exact
    assert torch.equal(feats("sakaguchi"), kur)
    assert torch.equal(feats("harmonic2"), kur)
    # nonzero inits: genuinely different physics
    sak = feats("sakaguchi", sakaguchi_alpha=math.pi / 4)
    har = feats("harmonic2", harmonic2_beta=0.5)
    assert not torch.allclose(sak, kur, atol=1e-4)
    assert not torch.allclose(har, kur, atol=1e-4)
    assert not torch.allclose(sak, har, atol=1e-4)


def test_randgraph_core_contract():
    #: disorder core — right shapes, seed-dependent, differs from circulant
    from harness import OscillatorField, RandGraphCore

    torch.manual_seed(0)
    rows = torch.rand(2, 30, 16) * 0.5
    def feats(seed, k=4):
        m = OscillatorField(channels=2, grid=16, n_classes=2, probe_seed=0, core="randgraph",
                     damping=0.3, spectral_clamp=1.0, gain=2.0, seed=seed, graph_k=k)
        return m.features(rows)
    f0, f1 = feats(0), feats(1)
    assert torch.isfinite(f0).all() and not torch.allclose(f0, f1, atol=1e-4)
    m_circ = OscillatorField(channels=2, grid=16, n_classes=2, probe_seed=0, core="phase",
                      damping=0.3, spectral_clamp=1.0, gain=2.0, seed=0)
    assert not torch.allclose(f0, m_circ.features(rows), atol=1e-3)
    core = RandGraphCore(channels=1, grid=16, dt=0.1, damping=0.3,
                         spectral_clamp=0.7, graph_k=8, seed=3)
    sv = torch.linalg.matrix_norm(core.W[0], ord=2)
    assert abs(float(sv) - 0.7) < 0.05  # spectral radius ~ clamp
    nnz = int((core.W[0] != 0).sum())
    assert nnz == 8 * 256  # graph_k nonzeros per oscillator
    # settle-read interface: forward_scan must accept a continuation state
    m = OscillatorField(channels=2, grid=16, n_classes=2, probe_seed=0, core="randgraph",
                 damping=0.3, spectral_clamp=1.0, gain=2.0, seed=0, graph_k=4)
    assert torch.isfinite(m.features_settle(rows)).all()

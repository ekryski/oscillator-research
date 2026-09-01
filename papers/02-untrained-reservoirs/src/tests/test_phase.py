"""Structural checks on the torus core: equivariance, stability, gradient flow.

Also the geometry builds (sheet/helix/cube/sphere): boundary correctness,
fft==matmul operator parity, spectral-clamp operator-norm bounds, and the
pre-registered tonotopic drive mappings."""

import math

import torch

from harness import PhaseCore

TWO_PI = 2 * math.pi


def _core(**kw) -> PhaseCore:
    defaults = dict(channels=4, grid=8, blocks=1, substeps=2, dt=0.1, coupling="kuramoto")
    defaults.update(kw)
    torch.manual_seed(0)
    return PhaseCore(**defaults)


def test_shift_equivariance():
    """The defining torus property: rolling state+drive rolls the output identically.
    (natural_freqs is made spatially constant — per-node frequencies are the one
    intentionally non-equivariant term.)"""
    core = _core()
    with torch.no_grad():
        core.blocks[0].natural_freqs.fill_(0.7)
    theta = torch.rand(2, 1, 4, 8, 8) * 6.28
    drive = torch.randn(2, 4, 8, 8)
    shift = (3, 5)

    state_out, _ = core.step_frame(theta, drive)
    theta_r = torch.roll(theta, shifts=shift, dims=(-2, -1))
    drive_r = torch.roll(drive, shifts=shift, dims=(-2, -1))
    state_out_r, _ = core.step_frame(theta_r, drive_r)
    assert torch.allclose(torch.roll(state_out, shifts=shift, dims=(-2, -1)), state_out_r, atol=1e-5)


def test_long_scan_stays_finite():
    core = _core()
    drives = torch.randn(1, 1000, 4, 8, 8)
    feats, state = core.forward_scan(drives)
    assert torch.isfinite(feats).all() and torch.isfinite(state).all()
    assert feats.abs().max() <= 1.0 + 1e-6  # readout is sin/cos-bounded


def test_gradient_reaches_kernel_and_freqs():
    core = _core()
    drives = torch.randn(2, 40, 4, 8, 8, requires_grad=True)
    feats, _ = core.forward_scan(drives)
    # NOTE: a symmetric loss like (sin^2 + cos^2).sum() is constant by identity and
    # would produce exactly-zero grads; use a random projection instead.
    (feats * torch.randn_like(feats)).sum().backward()
    blk = core.blocks[0]
    for name, p in (("kernel", blk.kernel), ("natural_freqs", blk.natural_freqs)):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"{name} got zero gradient"
    assert drives.grad is not None and drives.grad.abs().sum() > 0


def test_kuramoto_differs_from_forced():
    """The relative-phase term must actually change the dynamics."""
    kur = _core(coupling="kuramoto")
    frc = _core(coupling="forced")  # same seed -> same params
    drives = torch.randn(1, 10, 4, 8, 8)
    f1, _ = kur.forward_scan(drives)
    f2, _ = frc.forward_scan(drives)
    assert not torch.allclose(f1, f2)


def test_streaming_matches_batch_scan():
    """step_frame chained by hand == forward_scan (persistent state is the contract)."""
    core = _core()
    drives = torch.randn(1, 12, 4, 8, 8)
    feats_scan, final_scan = core.forward_scan(drives)
    state = core.init_state(1)
    coups = core.prepare_couplings()
    outs = []
    for i in range(12):
        state, feat = core.step_frame(state, drives[:, i], coups)
        outs.append(feat)
    feats_stream = torch.stack(outs, dim=1)
    assert torch.allclose(feats_scan, feats_stream, atol=1e-6)
    assert torch.allclose(final_scan, state, atol=1e-6)


def test_grad_ckpt_matches_full_bptt():
    core = _core()
    drives = torch.randn(1, 30, 4, 8, 8)

    core.train()
    feats_full, _ = core.forward_scan(drives)
    feats_full.sum().backward()
    g_full = core.blocks[0].kernel.grad.clone()
    core.zero_grad()

    feats_ckpt, _ = core.forward_scan(drives, grad_ckpt=8)
    assert torch.allclose(feats_full, feats_ckpt, atol=1e-6)
    feats_ckpt.sum().backward()
    assert torch.allclose(g_full, core.blocks[0].kernel.grad, atol=1e-5)


def test_coupling_impl_parity():
    """The fft and matmul (circulant) implementations compute the SAME operator —
    forward outputs and gradients must agree to float rounding."""
    fft_core = _core(coupling_impl="fft", damping=0.5, spectral_clamp=0.5)
    mm_core = _core(coupling_impl="matmul", damping=0.5, spectral_clamp=0.5)  # same seed -> same params
    drives = torch.randn(2, 40, 4, 8, 8)

    f1, s1 = fft_core.forward_scan(drives)
    f2, s2 = mm_core.forward_scan(drives)
    assert torch.allclose(f1, f2, atol=1e-4), (f1 - f2).abs().max()
    assert torch.allclose(s1, s2, atol=1e-4)

    (f1 * torch.linspace(0.5, 1.5, f1.numel()).view_as(f1)).sum().backward()
    (f2 * torch.linspace(0.5, 1.5, f2.numel()).view_as(f2)).sum().backward()
    g1, g2 = fft_core.blocks[0].kernel.grad, mm_core.blocks[0].kernel.grad
    # Gradients accumulate over 40 frames (magnitudes ~10^3): compare relatively.
    assert torch.allclose(g1, g2, rtol=1e-3, atol=1e-2), ((g1 - g2).abs() / (g1.abs() + 1e-6)).max()


def test_multiblock_forward_and_grad():
    core = _core(blocks=2)
    drives = torch.randn(2, 20, 4, 8, 8)
    feats, state = core.forward_scan(drives)
    assert state.shape == (2, 2, 4, 8, 8)
    feats.sum().backward()
    assert core.blocks[1].kernel.grad is not None
    assert core.mixers[0].weight.grad is not None


# ---------------------------------------------------------------------------
# geometry builds: sheet / helix / cube / sphere
# (pre-registered per-shape tonotopic mappings:
#
# ---------------------------------------------------------------------------

NEW_BOUNDARIES = ("sheet", "helix", "cube", "sphere")
# geometry expansion: the orientability pair + Parzival's diamond lattice
F069_BOUNDARIES = ("moebius", "klein", "diamond")


def _delta_core(boundary: str, grid: int, impl: str, taps: dict[tuple[int, int], float]) -> PhaseCore:
    """forced coupling, zero omega/damping, dt=1, one substep: a single step()
    reads the raw coupling-operator output on sin(theta) directly."""
    core = PhaseCore(channels=1, grid=grid, blocks=1, substeps=1, dt=1.0,
                     coupling="forced", damping=0.0, spectral_clamp=0.0,
                     coupling_impl=impl, boundary=boundary)
    with torch.no_grad():
        core.blocks[0].kernel.zero_()
        for (r, c), v in taps.items():
            core.blocks[0].kernel[0, r, c] = v
        core.blocks[0].natural_freqs.zero_()
    return core


def _one_step(core: PhaseCore, theta: torch.Tensor) -> torch.Tensor:
    """One step on a delta-kernel core -> the coupling operator's raw output.
    Circular difference: remainder() wraps a -1e-8 roundoff to ~2pi."""
    kf = core.prepare_couplings()
    new = core.blocks[0].step(theta[:, 0], torch.zeros_like(theta[:, 0]), kf[0], 1)
    return torch.remainder(new - theta[:, 0] + math.pi, TWO_PI) - math.pi


def test_sheet_boundary_open_both_axes():
    # Row-offset +1 tap: a pulse in the LAST row wraps to row 0 on the torus
    # and reaches NOTHING on the sheet (open edge); same story along columns;
    # interior propagation identical. (The cylinder contract, both axes.)
    g = 8
    for impl in ("fft", "matmul"):
        for axis, tap in (("row", (1, 0)), ("col", (0, 1))):
            outs = {}
            for boundary in ("torus", "sheet"):
                core = _delta_core(boundary, g, impl, {tap: 1.0})
                theta = torch.zeros(1, 1, 1, g, g)
                if axis == "row":
                    theta[0, 0, 0, g - 1, :] = 1.0  # pulse in the last row
                else:
                    theta[0, 0, 0, :, g - 1] = 1.0  # pulse in the last column
                outs[boundary] = _one_step(core, theta)[0, 0]
            edge = outs["torus"][0, :] if axis == "row" else outs["torus"][:, 0]
            assert edge.abs().sum() > 0.1  # torus wraps
            edge = outs["sheet"][0, :] if axis == "row" else outs["sheet"][:, 0]
            assert edge.abs().sum() < 1e-6  # sheet: open edge, nothing
            if axis == "row":
                assert torch.allclose(outs["torus"][1:], outs["sheet"][1:], atol=1e-6)
            else:
                assert torch.allclose(outs["torus"][:, 1:], outs["sheet"][:, 1:], atol=1e-6)


def test_helix_ring_seam_and_wrap():
    # Ring position p = row*G + col with a +1 circulant tap (kernel storage
    # [0, 1] = ring offset 1): the pulse CROSSES the row seam (position G-1 ->
    # position G = next row, col 0) and the ring CLOSES (position N-1 -> 0) —
    # both per the pre-registered circulant design (the seam couples the top
    # band to the bottom band; recorded in the core docstring).
    g = 8
    for impl in ("fft", "matmul"):
        core = _delta_core("helix", g, impl, {(0, 1): 1.0})
        for src, dst in ((g - 1, g), (g * g - 1, 0)):
            theta = torch.zeros(1, 1, 1, g, g)
            theta.view(-1)[src] = 1.0
            out = _one_step(core, theta)[0, 0].flatten()
            assert abs(out[dst].item() - math.sin(1.0)) < 1e-5, (impl, src)
            out[dst] = 0.0
            assert out.abs().max() < 1e-6  # nowhere else


def test_cube_all_axes_periodic_z_is_row():
    # (G, s, s) prism over the [G, G] storage: z = storage row (the tonotopic
    # axis), each row's G columns = one s x s slab (col = y*s + x). All three
    # axes are periodic (3-torus): z wraps band 15 -> band 0, y and x wrap
    # within the slab.
    g, s = 16, 4
    cases = (
        ((1, 0), (15, 11), (0, 11)),  # z+1 tap: z wraps 15 -> 0
        ((0, s), (3, 13), (3, 1)),    # y+1 tap: y wraps 3 -> 0 within the slab
        ((0, 1), (5, 11), (5, 8)),    # x+1 tap: x wraps 3 -> 0 within the slab
    )
    for impl in ("fft", "matmul"):
        for tap, (sr, sc), (dr, dc) in cases:
            core = _delta_core("cube", g, impl, {tap: 1.0})
            theta = torch.zeros(1, 1, 1, g, g)
            theta[0, 0, 0, sr, sc] = 1.0
            out = _one_step(core, theta)[0, 0]
            assert abs(out[dr, dc].item() - math.sin(1.0)) < 1e-5, (impl, tap)
            out[dr, dc] = 0.0
            assert out.abs().max() < 1e-6  # nowhere else


def test_sphere_open_poles_cos_weights_periodic_longitude():
    from harness.models.geometries import sphere_cos_weights
    g = 8
    w = sphere_cos_weights(g)
    assert w[0] < 0.5 * w[g // 2]  # near-pole ring genuinely down-weighted
    for impl in ("fft", "matmul"):
        core = _delta_core("sphere", g, impl, {(1, 0): 1.0})  # +1 latitude tap
        # open pole: a pulse in the LAST ring exits the top -> nothing anywhere
        theta = torch.zeros(1, 1, 1, g, g)
        theta[0, 0, 0, g - 1, 3] = 1.0
        assert _one_step(core, theta)[0, 0].abs().max() < 1e-6
        # metric weighting: out[r+1] = cos(lat_r) * sin(1.0) — the SOURCE ring's
        # area element scales its contribution
        for r in (0, g // 2):
            theta = torch.zeros(1, 1, 1, g, g)
            theta[0, 0, 0, r, 3] = 1.0
            out = _one_step(core, theta)[0, 0]
            assert abs(out[r + 1, 3].item() - w[r].item() * math.sin(1.0)) < 1e-5, (impl, r)
        # periodic longitude: a +1 column tap wraps the last column to column 0
        core = _delta_core("sphere", g, impl, {(0, 1): 1.0})
        theta = torch.zeros(1, 1, 1, g, g)
        theta[0, 0, 0, 3, g - 1] = 1.0
        out = _one_step(core, theta)[0, 0]
        assert abs(out[3, 0].item() - w[3].item() * math.sin(1.0)) < 1e-5


def test_new_boundaries_fft_equals_matmul():
    for boundary in NEW_BOUNDARIES + F069_BOUNDARIES:
        torch.manual_seed(3)
        theta = torch.rand(2, 1, 2, 16, 16) * TWO_PI
        drive = torch.randn(2, 2, 16, 16) * 0.1
        outs = []
        for impl in ("fft", "matmul"):
            torch.manual_seed(7)  # same kernel/omega draws
            core = PhaseCore(channels=2, grid=16, blocks=1, substeps=2, dt=0.1,
                             coupling="kuramoto", damping=0.1, spectral_clamp=0.5,
                             coupling_impl=impl, boundary=boundary)
            kf = core.prepare_couplings()
            outs.append(core.blocks[0].step(theta[:, 0], drive, kf[0], 2))
        assert torch.allclose(outs[0], outs[1], atol=1e-5), \
            f"{boundary}: fft vs matmul mismatch {(outs[0] - outs[1]).abs().max():.2e}"


def test_boundary_clamp_is_true_operator_norm_bound():
    # The dense matmul operator IS the coupling operator: its spectral norm may
    # never exceed the clamp, under any padding/embedding (cylinder/sheet/
    # sphere), reshaping (helix/cube), twisted double cover (moebius/klein —
    # whose cap is clamp/sqrt(2), the mirrored-copy extension carrying norm
    # sqrt(2)), or bipartite blocking (diamond). For the exact-norm shapes the
    # clamp is TIGHT (max |K-hat| is the operator norm — diamond included:
    # block-antidiagonal singular values are the block spectra), so a binding
    # clamp lands the norm right at the cap.
    clamp = 0.7
    for boundary in ("torus", "cylinder") + NEW_BOUNDARIES + F069_BOUNDARIES:
        torch.manual_seed(0)
        core = PhaseCore(channels=2, grid=16, blocks=1, spectral_clamp=clamp,
                         coupling_impl="matmul", boundary=boundary)
        with torch.no_grad():
            core.blocks[0].kernel.mul_(10.0)  # make the clamp bind hard
        norms = torch.linalg.matrix_norm(core.prepare_couplings()[0], ord=2)
        assert (norms <= clamp + 1e-3).all(), (boundary, norms)
        if boundary in ("torus", "helix", "cube", "diamond"):
            assert (norms >= clamp - 1e-2).all(), (boundary, norms)
        # non-vacuous: without the clamp the same kernel exceeds the cap
        torch.manual_seed(0)
        raw_core = PhaseCore(channels=2, grid=16, blocks=1, spectral_clamp=0.0,
                             coupling_impl="matmul", boundary=boundary)
        with torch.no_grad():
            raw_core.blocks[0].kernel.mul_(10.0)
        raw = torch.linalg.matrix_norm(raw_core.prepare_couplings()[0], ord=2)
        assert (raw > clamp).all(), (boundary, raw)


def test_new_boundaries_long_scan_stays_finite():
    for boundary in NEW_BOUNDARIES + F069_BOUNDARIES:
        torch.manual_seed(0)
        core = PhaseCore(channels=1, grid=16, blocks=1, substeps=2, boundary=boundary)
        drives = torch.randn(1, 1000, 1, 16, 16)
        feats, state = core.forward_scan(drives)
        assert feats.shape == (1, 1000, core.readout_dim)
        assert torch.isfinite(feats).all() and torch.isfinite(state).all(), boundary
        assert feats.abs().max() <= 1.0 + 1e-6  # readout stays sin/cos-bounded


def test_new_boundaries_grad_reaches_kernel_and_drive():
    for boundary in NEW_BOUNDARIES + F069_BOUNDARIES:
        torch.manual_seed(0)
        core = PhaseCore(channels=2, grid=16, blocks=1, substeps=2, boundary=boundary)
        drives = torch.randn(1, 25, 2, 16, 16, requires_grad=True)
        feats, _ = core.forward_scan(drives)
        (feats * torch.randn_like(feats)).sum().backward()
        blk = core.blocks[0]
        for name, p in (("kernel", blk.kernel), ("natural_freqs", blk.natural_freqs)):
            assert p.grad is not None and torch.isfinite(p.grad).all(), (boundary, name)
            assert p.grad.abs().sum() > 0, (boundary, name)
        assert drives.grad is not None and drives.grad.abs().sum() > 0, boundary


def test_helix_cube_sphere_translation_equivariance():
    # helix: rolling the RING rolls the output; cube: rolling any prism axis;
    # sphere: rolling LONGITUDE (the periodic axis — the cos(lat) weights are
    # row-only, so they commute with column rolls). omega made constant — the
    # per-node frequencies are the one intentionally non-equivariant term.
    def rolled_pair(boundary, roll_fn):
        torch.manual_seed(2)
        core = PhaseCore(channels=1, grid=16, blocks=1, substeps=1, boundary=boundary)
        with torch.no_grad():
            core.blocks[0].natural_freqs.fill_(0.7)
        theta = torch.rand(1, 1, 1, 16, 16) * TWO_PI
        drive = torch.randn(1, 1, 16, 16)
        out, _ = core.step_frame(theta, drive)
        out_r, _ = core.step_frame(roll_fn(theta), roll_fn(drive))
        return roll_fn(out), out_r

    a, b = rolled_pair("helix", lambda t: torch.roll(t.flatten(-2), 37, dims=-1).reshape(t.shape))
    assert torch.allclose(a, b, atol=1e-5)
    a, b = rolled_pair("cube", lambda t: torch.roll(
        t.reshape(*t.shape[:-2], 16, 4, 4), (3, 1, 2), dims=(-3, -2, -1)).reshape(t.shape))
    assert torch.allclose(a, b, atol=1e-5)
    a, b = rolled_pair("sphere", lambda t: torch.roll(t, 5, dims=-1))
    assert torch.allclose(a, b, atol=1e-5)


def test_drive_map_matches_preregistered_tonotopy():
    import sys
    sys.path.insert(0, ".")
    from harness import drive_map, rows_to_drive
    g = 16
    row_ids = torch.arange(g * g).view(g, g)
    for boundary in ("torus", "cylinder") + NEW_BOUNDARIES + F069_BOUNDARIES:
        m = drive_map(boundary, g)
        assert m.shape == (g, g) and m.dtype == torch.long
        # every pre-registered mapping lands on storage row b (chosen layouts)
        assert torch.equal(m, row_ids), boundary
    # helix: band b = ring positions G*b .. G*b+G-1 (16b..16b+15 — contiguous
    # quarter-octave arc; 4 bands x 16 positions = 64 = one turn = one octave)
    assert torch.equal(drive_map("helix", g)[3], torch.arange(48, 64))
    # cube: band b = the whole 4x4 z-slice b
    assert torch.equal(drive_map("cube", g)[5].view(4, 4), torch.arange(80, 96).view(4, 4))
    # diamond: band b = crystal layer b along a1 (odd b = the B sublattice of
    # run-plane b//2) — the 16 sites of one sublattice's 4x4 v-w sheet
    assert torch.equal(drive_map("diamond", g)[5].view(4, 4), torch.arange(80, 96).view(4, 4))
    # the harness broadcast puts a one-hot band EXACTLY on drive_map's sites
    rows = torch.zeros(1, 1, g)
    rows[0, 0, 5] = 1.0
    drive = rows_to_drive(rows, channels=2, gain=2.0)[0, 0, 0].flatten()
    assert torch.equal(drive.nonzero().flatten(), drive_map("sphere", g)[5])
    for bad in (lambda: drive_map("ring", g), lambda: drive_map("cube", 8),
                lambda: drive_map("diamond", 8)):
        try:
            bad()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_sphere_impl_attribute_records_lattice_approximation():
    core = _core(boundary="sphere")
    assert core.sphere_impl == "lattice"
    assert core.blocks[0].geometry.implementation == "lattice"
    assert _core().sphere_impl is None  # torus: not a sphere approximation


def test_cube_requires_perfect_square_grid():
    for boundary in ("cube", "diamond"):
        try:
            _core(boundary=boundary)  # default grid=8: cell dims undefined
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_kernel_support_rejected_on_nongrid_boundaries():
    for boundary, grid in (("helix", 8), ("cube", 16), ("sphere", 8), ("diamond", 16)):
        try:
            _core(boundary=boundary, grid=grid, kernel_support=1)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    # 2D signed offsets: mask valid (twisted pair included — keeps the
    # cylinder<->moebius / torus<->klein comparisons single-variable)
    for boundary in ("sheet", "moebius", "klein"):
        _core(boundary=boundary, kernel_support=1)


def test_winfree_open_boundary_exact_dc_and_parity():
    # Winfree on the boundaries uses the exact spatial K*1 DC response
    # (open edges / metric weights make it non-constant): fft and matmul must
    # agree, the scan stays finite, grads reach kernel + the winfree S params.
    torch.manual_seed(1)
    theta = torch.rand(2, 1, 2, 8, 8) * TWO_PI
    drive = torch.randn(2, 2, 8, 8) * 0.1
    outs = []
    for impl in ("fft", "matmul"):
        torch.manual_seed(6)
        core = PhaseCore(channels=2, grid=8, blocks=1, substeps=2, dt=0.1,
                         coupling="winfree", damping=0.1, spectral_clamp=0.5,
                         coupling_impl=impl, boundary="sheet")
        kf = core.prepare_couplings()
        outs.append(core.blocks[0].step(theta[:, 0], drive, kf[0], 2))
    assert torch.allclose(outs[0], outs[1], atol=1e-5), (outs[0] - outs[1]).abs().max()
    torch.manual_seed(6)
    core = PhaseCore(channels=2, grid=8, blocks=1, substeps=1, coupling="winfree",
                     boundary="sphere")
    drives = torch.randn(1, 30, 2, 8, 8, requires_grad=True)
    feats, _ = core.forward_scan(drives)
    assert torch.isfinite(feats).all()
    (feats * torch.randn_like(feats)).sum().backward()
    blk = core.blocks[0]
    assert blk.kernel.grad.abs().sum() > 0 and blk.winfree_s.grad.abs().sum() > 0


def test_multiblock_new_boundary_forward_and_grad():
    core = _core(blocks=2, boundary="helix")
    drives = torch.randn(2, 15, 4, 8, 8)
    feats, state = core.forward_scan(drives)
    assert state.shape == (2, 2, 4, 8, 8) and torch.isfinite(feats).all()
    feats.sum().backward()
    assert core.blocks[1].kernel.grad is not None


# ---------------------------------------------------------------------------
# geometry expansion: moebius / klein (the orientability pair) + diamond
# (Parzival's lattice). Pre-registered spec:
#
# ---------------------------------------------------------------------------


def test_moebius_seam_mirrors_frequency_axis():
    # The cylinder<->moebius single-variable pair: identical interior physics;
    # a pulse crossing the twisted column seam returns at the MIRRORED row —
    # the open/frequency axis flips, the non-orientability signature.
    g = 8
    for impl in ("fft", "matmul"):
        for boundary, dst_row in (("cylinder", 2), ("moebius", g - 1 - 2)):
            core = _delta_core(boundary, g, impl, {(0, 1): 1.0})  # column offset +1
            theta = torch.zeros(1, 1, 1, g, g)
            theta[0, 0, 0, 2, g - 1] = 1.0  # pulse at the seam column
            out = _one_step(core, theta)[0, 0]
            assert abs(out[dst_row, 0].item() - math.sin(1.0)) < 1e-5, (impl, boundary)
            out[dst_row, 0] = 0.0
            assert out.abs().max() < 1e-6, (impl, boundary)  # nowhere else
        # interior propagation identical between the pair members
        outs = {}
        for boundary in ("cylinder", "moebius"):
            core = _delta_core(boundary, g, impl, {(0, 1): 1.0})
            theta = torch.zeros(1, 1, 1, g, g)
            theta[0, 0, 0, 2, 3] = 1.0
            outs[boundary] = _one_step(core, theta)[0, 0]
        assert torch.allclose(outs["cylinder"], outs["moebius"], atol=1e-6), impl


def test_klein_row_seam_flips_columns_column_seam_plain():
    # The torus<->klein pair: a pulse crossing the ROW (frequency-axis) seam
    # returns at the MIRRORED column; the column seam stays an untwisted wrap.
    g = 8
    for impl in ("fft", "matmul"):
        for boundary, dst_col in (("torus", 5), ("klein", g - 1 - 5)):
            core = _delta_core(boundary, g, impl, {(1, 0): 1.0})  # row offset +1
            theta = torch.zeros(1, 1, 1, g, g)
            theta[0, 0, 0, g - 1, 5] = 1.0  # pulse in the last row
            out = _one_step(core, theta)[0, 0]
            assert abs(out[0, dst_col].item() - math.sin(1.0)) < 1e-5, (impl, boundary)
            out[0, dst_col] = 0.0
            assert out.abs().max() < 1e-6, (impl, boundary)
        # column wrap: untwisted (the flip acts on columns only when the ROW
        # seam is crossed) — and interior propagation matches the torus
        core = _delta_core("klein", g, impl, {(0, 1): 1.0})
        theta = torch.zeros(1, 1, 1, g, g)
        theta[0, 0, 0, 3, g - 1] = 1.0
        out = _one_step(core, theta)[0, 0]
        assert abs(out[3, 0].item() - math.sin(1.0)) < 1e-5, impl
        out[3, 0] = 0.0
        assert out.abs().max() < 1e-6, impl
        outs = {}
        for boundary in ("torus", "klein"):
            core = _delta_core(boundary, g, impl, {(1, 0): 1.0})
            theta = torch.zeros(1, 1, 1, g, g)
            theta[0, 0, 0, 3, 5] = 1.0
            outs[boundary] = _one_step(core, theta)[0, 0]
        assert torch.allclose(outs["torus"], outs["klein"], atol=1e-6), impl


def test_moebius_klein_twisted_translation_equivariance():
    # The twisted shapes' surviving symmetry: translation along the periodic
    # axis WITH the deck rule — entries that wrap the seam arrive mirrored.
    # It commutes only for kernels symmetric in the flipped axis's signed
    # offsets (a non-orientable venue has no global orientation for the
    # antisymmetric part — chart note in the core docstring): moebius rows are
    # symmetrized with the unpaired -G/2 offset zeroed (the open axis has no
    # +G/2 partner); klein columns are symmetrized (-G/2 = +G/2 mod G is
    # self-paired). omega made constant, as in the other equivariance tests.
    g = 16

    def troll_moebius(t: torch.Tensor, shift: int) -> torch.Tensor:
        r = torch.roll(t, shift, dims=-1).clone()
        r[..., :, :shift] = r[..., :, :shift].flip(-2)  # wrapped cols: row-mirrored
        return r

    def troll_klein(t: torch.Tensor, shift: int) -> torch.Tensor:
        r = torch.roll(t, shift, dims=-2).clone()
        r[..., :shift, :] = r[..., :shift, :].flip(-1)  # wrapped rows: col-mirrored
        return r

    conj = (-torch.arange(g)) % g
    for boundary, roll_fn in (("moebius", troll_moebius), ("klein", troll_klein)):
        torch.manual_seed(4)
        core = PhaseCore(channels=1, grid=g, blocks=1, substeps=1, boundary=boundary)
        with torch.no_grad():
            k = core.blocks[0].kernel
            if boundary == "moebius":
                k.copy_(0.5 * (k + k[:, conj, :]))
                k[:, g // 2, :] = 0.0
            else:
                k.copy_(0.5 * (k + k[:, :, conj]))
            core.blocks[0].natural_freqs.fill_(0.7)
        theta = torch.rand(1, 1, 1, g, g) * TWO_PI
        drive = torch.randn(1, 1, g, g)
        out, _ = core.step_frame(theta, drive)
        out_r, _ = core.step_frame(roll_fn(theta, 5), roll_fn(drive, 5))
        assert torch.allclose(roll_fn(out, 5), out_r, atol=1e-5), boundary


def test_diamond_tetrahedral_neighbors_and_bipartite():
    # Parzival's lattice: the four A<-B nearest-neighbor taps (bond vectors
    # tau, tau-a1, tau-a2, tau-a3 = kernel slots [0,0], [2,0], [0,4], [0,1])
    # land a B pulse on exactly its 4 tetrahedral A neighbors — adjacent
    # crystal layers only, u-axis wrap exercised. And the coupling is strictly
    # bipartite: with ANY kernel, an A-only pulse produces zero on A sites.
    g, s, nu = 16, 4, 8
    nn_taps = {(0, 0): 1.0, (2, 0): 1.0, (0, 4): 1.0, (0, 1): 1.0}
    u0, v0, w0 = 7, 1, 2  # u0 = 7 exercises the u-axis wrap (7+1 -> 0)
    expected = {(2 * ((u0 + du) % nu), ((v0 + dv) % s) * s + ((w0 + dw) % s))
                for du, dv, dw in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))}
    for impl in ("fft", "matmul"):
        core = _delta_core("diamond", g, impl, nn_taps)
        theta = torch.zeros(1, 1, 1, g, g)
        theta[0, 0, 0, 2 * u0 + 1, v0 * s + w0] = 1.0  # pulse on B(7, 1, 2)
        out = _one_step(core, theta)[0, 0]
        for r, c in expected:
            assert abs(out[r, c].item() - math.sin(1.0)) < 1e-5, (impl, r, c)
            out[r, c] = 0.0
        assert out.abs().max() < 1e-6, impl  # nowhere else: 4-coordination exactly
        # bipartite deadness: A-only pulse -> all-A silence, B response live
        core = _delta_core("diamond", g, impl, {})
        with torch.no_grad():
            torch.manual_seed(5)
            core.blocks[0].kernel.normal_()
        theta = torch.zeros(1, 1, 1, g, g)
        theta[0, 0, 0, 6, 9] = 1.0  # an A site (even layer)
        out = _one_step(core, theta)[0, 0]
        assert out[0::2].abs().max() < 1e-6, impl  # A hears nothing from A
        assert out[1::2].abs().sum() > 0.1, impl   # B hears A


def test_diamond_cell_translation_equivariance():
    # Rolling the storage by whole primitive-run steps commutes with the
    # coupling: +2 rows = +1 run along a1 (layer l -> l+2, sublattice kept);
    # +4 flat columns = +1 run along a2 (v -> v+1); +1 within each 4-column
    # slab = +1 run along a3 (w -> w+1).
    torch.manual_seed(2)
    core = PhaseCore(channels=1, grid=16, blocks=1, substeps=1, boundary="diamond")
    with torch.no_grad():
        core.blocks[0].natural_freqs.fill_(0.7)
    theta = torch.rand(1, 1, 1, 16, 16) * TWO_PI
    drive = torch.randn(1, 1, 16, 16)
    rolls = (
        lambda t: torch.roll(t, 2, dims=-2),
        lambda t: torch.roll(t, 4, dims=-1),
        lambda t: torch.roll(t.reshape(*t.shape[:-2], 16, 4, 4), 1, dims=-1
                             ).reshape(t.shape),
    )
    out, _ = core.step_frame(theta, drive)
    for roll_fn in rolls:
        out_r, _ = core.step_frame(roll_fn(theta), roll_fn(drive))
        assert torch.allclose(roll_fn(out), out_r, atol=1e-5)


# ---------------------------------------------------------------------------
# quadrature Adler drive (gate build 1b core hook)
# ---------------------------------------------------------------------------


def test_quad_drive_adler_fixed_point():
    # Bare oscillators (zero kernel/omega/damping): a quad drive whose phi_bb
    # equals theta exerts ZERO torque (sin(phi - theta) = 0, the Adler fixed
    # point); phi = theta + pi/2 exerts torque exactly +A (theta advances dt*A).
    core = PhaseCore(channels=1, grid=8, blocks=1, substeps=1, dt=0.1,
                     coupling="kuramoto", damping=0.0, spectral_clamp=0.0)
    with torch.no_grad():
        core.blocks[0].kernel.zero_()
        core.blocks[0].natural_freqs.zero_()
    torch.manual_seed(0)
    theta = torch.rand(2, 1, 1, 8, 8) * TWO_PI
    amp = torch.rand(2, 1, 8, 8) * 2
    aligned = torch.stack((amp * torch.cos(theta[:, 0]), amp * torch.sin(theta[:, 0])), dim=-1)
    new, _ = core.step_frame(theta, torch.zeros(2, 1, 8, 8), drive_quad=aligned)
    d = torch.remainder(new - theta + math.pi, TWO_PI) - math.pi
    assert d.abs().max() < 1e-6  # fixed point: no motion
    quarter = torch.stack((-amp * torch.sin(theta[:, 0]), amp * torch.cos(theta[:, 0])), dim=-1)
    new_q, _ = core.step_frame(theta, torch.zeros(2, 1, 8, 8), drive_quad=quarter)
    d_q = torch.remainder(new_q - theta + math.pi, TWO_PI) - math.pi
    assert torch.allclose(d_q[:, 0], 0.1 * amp, atol=1e-5)  # dt * A exactly


def test_quad_drive_scan_finite_and_zero_quad_matches_none():
    core = _core(damping=0.5, spectral_clamp=0.5)
    zeros = torch.zeros(1, 1000, 4, 8, 8)
    quad = torch.randn(1, 1000, 4, 8, 8, 2)
    feats, state = core.forward_scan(zeros, drives_quad=quad)
    assert torch.isfinite(feats).all() and torch.isfinite(state).all()
    # frozen-control guard: an all-zero quad drive is bit-identical to None
    drives = torch.randn(1, 40, 4, 8, 8) * 0.5
    f_none, s_none = core.forward_scan(drives)
    f_zero, s_zero = core.forward_scan(drives, drives_quad=torch.zeros(1, 40, 4, 8, 8, 2))
    assert torch.equal(f_none, f_zero) and torch.equal(s_none, s_zero)


def test_quad_drive_grad_reaches_kernel_and_ckpt_parity():
    core = _core()
    core.train()
    quad = torch.randn(2, 30, 4, 8, 8, 2, requires_grad=True)
    zeros = torch.zeros(2, 30, 4, 8, 8)
    feats, _ = core.forward_scan(zeros, drives_quad=quad)
    proj = torch.randn_like(feats)
    (feats * proj).sum().backward()
    blk = core.blocks[0]
    for name, p in (("kernel", blk.kernel), ("natural_freqs", blk.natural_freqs)):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"{name} got zero gradient through the quad path"
    assert quad.grad is not None and quad.grad.abs().sum() > 0
    g_full = blk.kernel.grad.clone()
    core.zero_grad()
    feats_ckpt, _ = core.forward_scan(zeros, drives_quad=quad, grad_ckpt=8)
    assert torch.allclose(feats, feats_ckpt, atol=1e-6)
    (feats_ckpt * proj).sum().backward()
    assert torch.allclose(g_full, blk.kernel.grad, atol=1e-5)

"""Contracts for `harness.models.geometries` — the nine lattice venues.

Geometry is a factor in the paper, so its implementation has to be exactly
what the factor claims: only the wrap rule at the lattice edge changes, the
parameter budget stays matched, and the tonotopic map stays pinned. On top of
that, the two coupling implementations have to be one operator, so that
choosing the faster one per device is an engineering decision and never a
scientific one.
"""

import math
import sys

import pytest
import torch

sys.path.insert(0, ".")
from harness.models.geometries import (
    BOUNDARIES,
    GEOMETRIES,
    build_geometry,
    cube_dims,
    diamond_dims,
    drive_map,
    sphere_cos_weights,
    sphere_latitudes,
)
from harness.models.phase import PhaseBlock, PhaseCore

TWO_PI = 2 * math.pi


def test_cylinder_boundary_does_not_wrap_rows():
    # delta kernel at row-offset +1: a pulse in the LAST row reaches row 0 on the
    # torus (wrap) and reaches NOTHING on the cylinder (open edge).
    for impl in ("fft", "matmul"):
        outs = {}
        for boundary in ("torus", "cylinder"):
            core = PhaseCore(channels=1, grid=8, blocks=1, substeps=1, dt=1.0,
                             coupling="forced", damping=0.0, spectral_clamp=0.0,
                             coupling_impl=impl, boundary=boundary)
            with torch.no_grad():
                core.blocks[0].kernel.zero_()
                core.blocks[0].kernel[0, 1, 0] = 1.0  # neighbor one row below sways me
                core.blocks[0].natural_freqs.zero_()
            theta = torch.zeros(1, 1, 1, 8, 8)
            theta[0, 0, 0, 7, :] = 1.0  # pulse in the last row
            kf = core.prepare_couplings()
            new = core.blocks[0].step(theta[:, 0], torch.zeros(1, 1, 8, 8), kf[0], 1)
            # circular difference: remainder() wraps a -1e-8 roundoff to ~2pi
            outs[boundary] = torch.remainder(new - theta[:, 0] + math.pi, TWO_PI) - math.pi
        assert outs["torus"][0, 0, 0].abs().sum() > 0.1     # wrapped to row 0
        assert outs["cylinder"][0, 0, 0].abs().sum() < 1e-6  # open edge: nothing
        # interior propagation identical under both boundaries
        assert torch.allclose(outs["torus"][0, 0, 1:], outs["cylinder"][0, 0, 1:], atol=1e-6)


def test_cylinder_fft_equals_matmul():
    torch.manual_seed(3)
    theta = torch.rand(2, 1, 2, 8, 8) * TWO_PI
    drive = torch.randn(2, 2, 8, 8) * 0.1
    outs = []
    for impl in ("fft", "matmul"):
        torch.manual_seed(7)  # same kernel/omega draws
        core = PhaseCore(channels=2, grid=8, blocks=1, substeps=2, dt=0.1,
                         coupling="kuramoto", damping=0.1, spectral_clamp=0.5,
                         coupling_impl=impl, boundary="cylinder")
        kf = core.prepare_couplings()
        outs.append(core.blocks[0].step(theta[:, 0], drive, kf[0], 2))
    assert torch.allclose(outs[0], outs[1], atol=1e-5), \
        f"fft vs matmul cylinder mismatch: {(outs[0]-outs[1]).abs().max():.2e}"


def test_every_declared_boundary_is_buildable_and_parameter_matched():
    """The nine shapes reuse one [C, G, G] storage: only the wrap rule differs,
    so the parameter budget is matched by construction rather than by tuning."""
    counts = {}
    for boundary in BOUNDARIES:
        torch.manual_seed(0)
        core = PhaseCore(channels=2, grid=16, blocks=1, substeps=1, dt=0.1,
                         coupling="kuramoto", damping=0.3, spectral_clamp=1.0,
                         seed=0, boundary=boundary)
        counts[boundary] = sum(p.numel() for p in core.parameters() if p.requires_grad)
    assert len(set(counts.values())) == 1, counts
    assert set(counts) == set(BOUNDARIES)


def test_drive_map_puts_band_b_on_storage_row_b_for_every_shape():
    """The pre-registered tonotopy. `rows_to_drive` broadcasts band b onto row
    b; this is the contract that makes that broadcast geometry-correct."""
    grid = 16
    expected = torch.arange(grid * grid).view(grid, grid)
    for boundary in BOUNDARIES:
        assert torch.equal(drive_map(boundary, grid), expected), boundary


def test_drive_map_rejects_unknown_boundaries():
    with pytest.raises(ValueError, match="unknown boundary"):
        drive_map("hyperboloid", 16)


def test_cube_and_diamond_require_the_grids_their_layouts_assume():
    assert cube_dims(16) == (16, 4, 4)
    assert diamond_dims(16) == (8, 4, 4)
    with pytest.raises(ValueError, match="perfect-square"):
        cube_dims(12)
    with pytest.raises(ValueError, match="perfect-square"):
        diamond_dims(12)
    with pytest.raises(ValueError, match="perfect-square"):
        diamond_dims(9)  # odd: no A/B sublattice split


def test_sphere_latitudes_span_the_globe_with_open_poles():
    lat = sphere_latitudes(16)
    assert lat.shape == (16,)
    assert torch.all(lat[1:] > lat[:-1]), "south to north, strictly increasing"
    assert lat.min() > -math.pi / 2 and lat.max() < math.pi / 2, "poles stay open"
    assert torch.allclose(lat, -lat.flip(0), atol=1e-6), "symmetric about the equator"


def test_sphere_weights_are_the_area_element_and_downweight_the_poles():
    w = sphere_cos_weights(16)
    assert torch.all(w > 0) and torch.all(w <= 1)
    assert torch.allclose(w, torch.cos(sphere_latitudes(16)))
    assert w[0] < w[8] and w[-1] < w[8], "near-pole rings oversample, so weigh less"


def test_periodic_shapes_wrap_and_open_shapes_do_not():
    """A pulse in the last row reaches row 0 on a shape with a periodic row
    axis and reaches nothing on one with an open row edge."""
    grid = 8
    reach = {}
    for boundary in ("torus", "cylinder", "sheet", "klein"):
        blk = PhaseBlock(channels=1, grid=grid, dt=1.0, coupling="forced", damping=0.0,
                         spectral_clamp=0.0, coupling_impl="matmul", boundary=boundary)
        with torch.no_grad():
            blk.kernel.zero_()
            blk.kernel[0, 1, 0] = 1.0          # a delta at row-offset +1
            blk.natural_freqs.zero_()
        theta = torch.zeros(1, 1, grid, grid)
        theta[0, 0, grid - 1, :] = math.pi / 2  # pulse in the LAST row
        coup = blk.prepare_coupling()
        out = blk.step(theta, torch.zeros_like(theta), coup, substeps=1)
        reach[boundary] = out[0, 0, 0].abs().max().item()
    # float32 FFT round-trip leaves ~1e-8 of noise, so compare magnitudes
    assert min(reach["torus"], reach["klein"]) > 1e-3, f"periodic rows wrap: {reach}"
    assert max(reach["cylinder"], reach["sheet"]) < 1e-5, f"open rows do not: {reach}"


# --- the two implementations are one operator -------------------------------

@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_spectral_and_dense_paths_are_the_same_operator(boundary):
    """FFT and dense matmul must agree on every venue.

    This is what lets the runner pick whichever is faster on the device it
    finds — measured, the choice is worth 3x either way depending on hardware —
    without that choice being a silent scientific variable.
    """
    outs = {}
    for impl in ("fft", "matmul"):
        torch.manual_seed(0)
        core = PhaseCore(channels=2, grid=16, blocks=1, substeps=2, dt=0.1,
                         coupling="kuramoto", damping=0.3, spectral_clamp=1.0,
                         coupling_impl=impl, seed=7, boundary=boundary)
        torch.manual_seed(1)
        outs[impl], _ = core.forward_scan(torch.randn(2, 5, 2, 16, 16) * 0.3)
    diff = (outs["fft"] - outs["matmul"]).abs().max().item()
    scale = outs["fft"].abs().max().item()
    assert diff < 1e-4 * scale, f"{boundary}: paths diverge by {diff:.2e}"


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_forward_passes_are_bit_identical_across_runs(boundary):
    """Every number the paper reports comes from a FORWARD pass on a frozen
    field, so forward determinism is the replication claim. Backward is a
    different story — see test_probe.py — because the dense path's gather
    accumulates in an unpinned order."""
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        core = PhaseCore(channels=2, grid=16, blocks=1, substeps=2, dt=0.1,
                         coupling="kuramoto", damping=0.3, spectral_clamp=1.0,
                         seed=7, boundary=boundary)
        torch.manual_seed(1)
        with torch.no_grad():
            feats, _ = core.forward_scan(torch.randn(2, 5, 2, 16, 16) * 0.3)
        outs.append(feats)
    assert torch.equal(outs[0], outs[1]), f"{boundary}: forward is not reproducible"


@pytest.mark.parametrize("coupling", ["kuramoto", "forced", "sakaguchi", "harmonic2", "winfree"])
def test_every_coupling_law_runs_on_every_venue(coupling):
    """The matrix crosses coupling law with geometry, so no combination may be
    quietly broken — a missing run would silently bias a family average."""
    for boundary in BOUNDARIES:
        torch.manual_seed(0)
        core = PhaseCore(channels=2, grid=16, blocks=1, substeps=1, dt=0.1,
                         coupling=coupling, damping=0.3, spectral_clamp=1.0,
                         seed=0, boundary=boundary, sakaguchi_alpha=0.785,
                         harmonic2_beta=0.5)
        feats, _ = core.forward_scan(torch.randn(1, 4, 2, 16, 16) * 0.3)
        assert torch.isfinite(feats).all(), f"{boundary} x {coupling}"


def test_only_the_twisted_venues_correct_their_spectral_clamp():
    """The mirrored double-cover extension has operator norm sqrt(2), so
    moebius and klein must cap lower to keep the clamp a true bound. Every
    other venue's max |K-hat| already bounds its operator."""
    for name in BOUNDARIES:
        geom = build_geometry(name, 16)
        expected = math.sqrt(2.0) if name in ("moebius", "klein") else 1.0
        assert geom.clamp_factor == pytest.approx(expected), name


def test_geometry_registry_is_complete_and_self_describing():
    assert set(GEOMETRIES) == set(BOUNDARIES)
    for name, cls in GEOMETRIES.items():
        assert cls.name == name, f"{cls.__name__} disagrees with its registry key"
        assert cls.frequency_axis, f"{name} must declare its tonotopic axis"
    with pytest.raises(ValueError, match="unknown boundary"):
        build_geometry("hyperboloid", 16)

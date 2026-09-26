"""Contracts for `harness.models.geometries`: the nine lattice geometries at every lattice size.

Geometry is a factor in the paper, so only the wrap rule at the lattice edge
may change between geometries: the parameter count stays matched and band b
stays on row b. The two coupling implementations must be one operator, so
choosing the faster one per device is never a scientific variable. Paper 03
adds lattices of 8, 32, 64 and 128 rows, a cube whose rows are cut into the
most nearly square slab the lattice allows, and paper 02's coil and cochlea,
defined at every lattice size by shares of the coil's length.
"""

import math

import pytest
import torch

from harness.models.geometries import (
    BOUNDARIES,
    GEOMETRIES,
    build_geometry,
    cube_dims,
    sphere_cos_weights,
    sphere_latitudes,
)
from harness.models.phase import PhaseBlock, PhaseCore

TWO_PI = 2 * math.pi


def _last_row_reaches_row_zero(boundary: str, grid: int = 8) -> float:
    """A delta kernel at row offset +1 and a pulse in the last row: how far row 0 moves in one step."""
    blk = PhaseBlock(channels=1, grid=grid, dt=1.0, coupling="kuramoto", damping=0.0, spectral_clamp=0.0,
                     coupling_impl="matmul", boundary=boundary)
    with torch.no_grad():
        blk.kernel.zero_()
        blk.kernel[0, 1, 0] = 1.0
        blk.natural_freqs.zero_()
    theta = torch.zeros(1, 1, grid, grid)
    theta[0, 0, grid - 1, :] = math.pi / 2
    out = blk.step(theta, torch.zeros_like(theta), blk.prepare_coupling(), substeps=1)
    return out[0, 0, 0].abs().max().item()


def test_periodic_rows_wrap_and_open_rows_do_not():
    reach = {b: _last_row_reaches_row_zero(b) for b in ("torus", "cylinder", "sheet", "sphere")}
    assert reach["torus"] > 1e-3, reach
    assert max(reach["cylinder"], reach["sheet"], reach["sphere"]) < 1e-5, reach


def test_the_nine_geometries_are_paper_02s_and_share_one_parameter_count():
    assert BOUNDARIES == ("torus", "cylinder", "sheet", "helix", "cube", "sphere", "coil", "cochlea", "cochlea-matched")
    counts = set()
    for boundary in BOUNDARIES:
        torch.manual_seed(0)
        core = PhaseCore(channels=2, grid=16, damping=0.3, spectral_clamp=1.0, boundary=boundary)
        counts.add(sum(p.numel() for p in core.parameters()))
    assert counts == {2 * 2 * 16 * 16}


def test_the_cube_is_paper_02s_where_the_lattice_is_a_square_and_the_nearest_slab_elsewhere():
    assert cube_dims(16) == (16, 4, 4) and cube_dims(64) == (64, 8, 8)
    assert (cube_dims(8), cube_dims(32), cube_dims(128)) == ((8, 2, 4), (32, 4, 8), (128, 8, 16))
    for grid in (8, 16, 32, 64, 128):
        _, a, b = cube_dims(grid)
        assert a * b == grid and a <= b


def test_sphere_latitudes_have_open_poles_and_area_weights():
    lat, w = sphere_latitudes(16), sphere_cos_weights(16)
    assert torch.all(lat[1:] > lat[:-1]) and lat.min() > -math.pi / 2 and lat.max() < math.pi / 2
    assert torch.allclose(w, torch.cos(lat)) and w[0] < w[8]


@pytest.mark.parametrize("grid", [8, 16, 32])
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_spectral_and_dense_paths_are_the_same_operator(boundary, grid):
    outs = {}
    for impl in ("fft", "matmul"):
        torch.manual_seed(0)
        core = PhaseCore(channels=2, grid=grid, substeps=2, damping=0.3, spectral_clamp=1.0,
                         coupling_impl=impl, seed=7, boundary=boundary)
        torch.manual_seed(1)
        outs[impl], _ = core.forward_scan(torch.randn(2, 5, 2, grid, grid) * 0.3)
    diff = (outs["fft"] - outs["matmul"]).abs().max().item()
    assert diff < 1e-4, f"{boundary} at {grid}: paths diverge by {diff:.2e}"


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_forward_passes_are_bit_identical_across_runs(boundary):
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        core = PhaseCore(channels=2, grid=16, substeps=2, damping=0.3, spectral_clamp=1.0, seed=7,
                         boundary=boundary)
        torch.manual_seed(1)
        with torch.no_grad():
            outs.append(core.forward_scan(torch.randn(2, 5, 2, 16, 16) * 0.3)[0])
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("grid", [8, 64])
@pytest.mark.parametrize("coupling", ["kuramoto", "sakaguchi", "harmonic2", "winfree"])
def test_every_coupling_function_runs_on_every_geometry_and_lattice(coupling, grid):
    for boundary in BOUNDARIES:
        torch.manual_seed(0)
        core = PhaseCore(channels=1, grid=grid, damping=0.3, spectral_clamp=1.0, boundary=boundary,
                         coupling=coupling, sakaguchi_alpha=0.785, harmonic2_beta=0.5)
        feats, _ = core.forward_scan(torch.randn(1, 3, 1, grid, grid) * 0.3)
        assert torch.isfinite(feats).all() and feats.abs().max() <= 1 + 1e-6, f"{boundary} x {coupling}"


def test_auto_couples_densely_where_that_is_faster_and_builds_no_dense_operator_otherwise():
    from harness.models.phase import auto_impl
    assert [auto_impl("cpu", g) for g in (8, 16, 32, 64, 128)] == ["matmul", "matmul", "fft", "fft", "fft"]
    assert [auto_impl("mps", g) for g in (8, 16, 32, 64, 128)] == ["matmul"] * 4 + ["fft"]
    assert {auto_impl("cuda", g) for g in (8, 16, 128)} == {"fft"}
    for grid in (32, 64, 128):
        blk = PhaseBlock(channels=1, grid=grid, spectral_clamp=1.0)
        blk.prepare_coupling()
        assert blk.coupling_impl == "fft" and blk._circ is None
    blk = PhaseBlock(channels=1, grid=16, spectral_clamp=1.0)
    blk.prepare_coupling()
    assert blk.coupling_impl == "matmul" and blk._circ is not None


def test_geometry_registry_is_complete_and_self_describing():
    assert set(GEOMETRIES) == set(BOUNDARIES)
    for name, cls in GEOMETRIES.items():
        assert cls.name == name and cls.frequency_axis
    with pytest.raises(ValueError, match="unknown boundary"):
        build_geometry("hyperboloid", 16)


# --- the coil and the cochlea, at every lattice size ---------------------------------------

GRIDS = (8, 16, 32, 64, 128)
COILS = ("coil", "cochlea", "cochlea-matched")


def _coil_response(boundary: str, taps: dict[int, float], source: int, grid: int) -> torch.Tensor:
    """One unscaled coupling step from a pulse at coil position `source`, every other oscillator at phase 0:
    the flat response [N]. With the Kuramoto function each site i moves by the tap at offset i - source,
    times the cochlea's curvature weight at i."""
    n = grid * grid
    blk = PhaseBlock(channels=1, grid=grid, dt=1.0, coupling="kuramoto", damping=0.0, spectral_clamp=0.0,
                     coupling_impl="matmul" if grid <= 32 else "fft", boundary=boundary)
    with torch.no_grad():
        blk.kernel.zero_()
        for offset, value in taps.items():                 # flat tap index: offset, or n + offset if negative
            blk.kernel.view(1, n)[0, offset % n] = value
        blk.natural_freqs.zero_()
    theta = torch.zeros(1, 1, grid, grid)
    theta.view(1, 1, n)[0, 0, source] = math.pi / 2
    with torch.no_grad():
        out = blk.step(theta, torch.zeros_like(theta), blk.prepare_coupling(), substeps=1)
    return out.view(n) - theta.view(n)


@pytest.mark.parametrize("grid", GRIDS)
def test_the_coil_is_open_where_the_helix_closes_at_every_lattice(grid):
    n = grid * grid
    for boundary, closes in (("helix", True), *((c, False) for c in COILS)):
        reach = _coil_response(boundary, {1: 1.0}, source=n - 1, grid=grid)[0].abs().item()   # base end -> apex end
        assert (reach > 1e-3) == closes, f"{boundary} at {grid}: {reach}"


@pytest.mark.parametrize("grid", GRIDS)
def test_the_kernels_taps_reach_one_turn_and_up_to_two_along_the_coil(grid):
    n = grid * grid
    turn = build_geometry("coil", grid).turn
    assert turn == n // 4 and turn == grid * (grid // 4)             # four turns, G / 4 rows to a turn
    source = n // 2
    near = _coil_response("coil", {turn: 1.0, -turn: 1.0}, source, grid)
    assert near[source + turn].item() == pytest.approx(1.0, abs=1e-4)
    assert near[source - turn].item() == pytest.approx(1.0, abs=1e-4)
    # the kernel's G^2 taps are the signed offsets -N/2 .. N/2 - 1: two turns toward the apex, a site short
    # of two toward the base
    base = _coil_response("coil", {n // 2 - 1: 1.0}, 0, grid)
    apex = _coil_response("coil", {-(n // 2): 1.0}, n - 1, grid)
    assert base[n // 2 - 1].item() == pytest.approx(1.0, abs=1e-4)
    assert apex[n // 2 - 1].item() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("grid", GRIDS)
def test_the_cochlea_carries_influence_toward_the_apex_and_the_coil_both_ways_at_every_lattice(grid):
    # a symmetric kernel: the coil passes it on equally; the cochlea 3:1 toward the apex, times the
    # curvature weights of the two receiving sites
    source = grid * grid // 2 + 3
    for boundary in COILS:
        w = build_geometry(boundary, grid).curvature(torch.zeros(1))
        ratio = 1.0 if w is None else 3.0 * (w[source - 1] / w[source + 1]).item()
        r = _coil_response(boundary, {1: 1.0, -1: 1.0}, source=source, grid=grid)
        toward_apex, toward_base = r[source - 1].item(), r[source + 1].item()   # lower positions are lower bands
        assert toward_apex / toward_base == pytest.approx(ratio, rel=1e-4), (boundary, grid)


@pytest.mark.parametrize("grid", GRIDS)
def test_the_cochlea_couples_most_strongly_at_the_apex_at_every_lattice(grid):
    w = build_geometry("cochlea", grid).curvature(torch.zeros(1))
    assert w.shape == (grid * grid,) and w[0].item() == pytest.approx(1.0) and w[-1].item() == pytest.approx(0.25)
    assert torch.all(w[1:] < w[:-1]) and build_geometry("coil", grid).curvature(torch.zeros(1)) is None
    # the matched control: the same shape of weighting, at the coil's average coupling, about 1.85 at the apex
    m = build_geometry("cochlea-matched", grid).curvature(torch.zeros(1))
    assert m.mean().item() == pytest.approx(1.0) and torch.allclose(m / m[0], w / w[0])
    assert 1.8 < m[0].item() < 1.9


def test_the_coil_needs_a_lattice_it_can_wind_four_times():
    for grid in GRIDS:
        build_geometry("cochlea", grid)
    with pytest.raises(ValueError, match="multiple of 4"):
        build_geometry("coil", 6)


@pytest.mark.parametrize("grid", GRIDS)
def test_the_coils_ceiling_bounds_its_operator_and_its_forward_pass_stays_on_the_circle(grid):
    for boundary in COILS:
        torch.manual_seed(0)
        blk = PhaseBlock(2, grid, spectral_clamp=1.0, boundary=boundary, coupling_impl="fft")
        spectrum = blk.prepare_coupling().abs().flatten(1).amax(dim=1)
        assert torch.allclose(spectrum, torch.ones(2), atol=1e-5), (boundary, grid)
        core = PhaseCore(channels=1, grid=grid, damping=0.3, spectral_clamp=1.0, boundary=boundary,
                         coupling="winfree")
        feats, _ = core.forward_scan(torch.randn(1, 3, 1, grid, grid) * 0.3)
        assert torch.isfinite(feats).all() and feats.abs().max() <= 1 + 1e-6

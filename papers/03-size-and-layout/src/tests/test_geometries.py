"""Contracts for `harness.models.geometries`: the six lattice geometries at every lattice size.

Geometry is a factor in the paper, so only the wrap rule at the lattice edge
may change between geometries: the parameter count stays matched and band b
stays on row b. The two coupling implementations must be one operator, so
choosing the faster one per device is never a scientific variable. Paper 03
adds lattices of 8, 32, 64 and 128 rows, and a cube whose rows are cut into
the most nearly square slab the lattice allows.
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


def test_the_six_geometries_are_paper_02s_and_share_one_parameter_count():
    assert BOUNDARIES == ("torus", "cylinder", "sheet", "helix", "cube", "sphere")
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

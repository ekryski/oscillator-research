"""Lattice venues for the oscillator field — one module per geometry.

Nine `boundary` values reuse one [C, G, G] parameter and state storage and
re-interpret it as different venues. The stadium picture: the seats never move,
only the seating chart changes. Everything here is derived, deterministic, and
parameter-free, so varying the geometry varies exactly one thing.

Pre-registered tonotopy. Every internal layout was chosen so the shape's
declared frequency axis lands on STORAGE ROW b for input band b, which is what
makes the harness's one broadcast rule (band b -> row b, all channels and
columns) geometry-correct everywhere:

| boundary | declared frequency axis              | band b drives                    |
|----------|--------------------------------------|----------------------------------|
| torus    | grid rows                            | row b                            |
| cylinder | the open axis (rows)                 | row b                            |
| sheet    | axis 0 (rows; both axes open)        | row b                            |
| helix    | along the coil, one octave per turn  | ring positions G*b..G*b+G-1      |
| cube     | z axis (fully periodic)              | z-slice b (an s x s slab)        |
| sphere   | latitude, south = low -> north       | latitude ring b                  |
| moebius  | the open axis (rows), as cylinder    | row b                            |
| klein    | grid rows, as torus                  | row b                            |
| diamond  | the a1 cell axis, A/B interleaved    | crystal layer b                  |

`drive_map` states that mapping explicitly, so callers and tests pin the
contract rather than relying on the layout coincidence silently.
"""

from __future__ import annotations

import torch

from harness.models.geometries.base import Geometry, PlanarGeometry
from harness.models.geometries.cube import Cube, cube_dims
from harness.models.geometries.cylinder import Cylinder
from harness.models.geometries.diamond import Diamond, diamond_dims
from harness.models.geometries.helix import Helix
from harness.models.geometries.sheet import Sheet
from harness.models.geometries.sphere import Sphere, sphere_cos_weights, sphere_latitudes
from harness.models.geometries.torus import Torus
from harness.models.geometries.twisted import TWIST_NORM_FACTOR, Klein, Moebius

#: name -> class, in the order the paper's tables list them
GEOMETRIES: dict[str, type[Geometry]] = {
    g.name: g for g in (Torus, Cylinder, Sheet, Helix, Cube, Sphere, Moebius, Klein, Diamond)
}
BOUNDARIES = tuple(GEOMETRIES)


def build_geometry(boundary: str, grid: int) -> Geometry:
    """Instantiate one venue, validating that it can exist at this grid."""
    if boundary not in GEOMETRIES:
        raise ValueError(f"unknown boundary '{boundary}' — expected one of {BOUNDARIES}")
    return GEOMETRIES[boundary](grid)


def drive_map(boundary: str, grid: int) -> torch.Tensor:
    """The pre-registered tonotopic drive mapping: [G, G] whose row b holds the
    G flat oscillator indices (into the row-major [G*G] flattening) that input
    band b drives.

    Every venue's answer is storage row b BY CONSTRUCTION of its internal
    layout — callers should treat this function, not that identity, as the
    contract, so a future layout change has one place to stay honest.
    """
    build_geometry(boundary, grid)  # validates the grid for this venue
    return torch.arange(grid * grid, dtype=torch.long).view(grid, grid)


__all__ = [
    "BOUNDARIES",
    "GEOMETRIES",
    "TWIST_NORM_FACTOR",
    "Cube",
    "Cylinder",
    "Diamond",
    "Geometry",
    "Helix",
    "Klein",
    "Moebius",
    "PlanarGeometry",
    "Sheet",
    "Sphere",
    "Torus",
    "build_geometry",
    "cube_dims",
    "diamond_dims",
    "drive_map",
    "sphere_cos_weights",
    "sphere_latitudes",
]

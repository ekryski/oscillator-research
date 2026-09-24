"""The lattice geometries, one module per geometry.

The six geometries of paper 02 reuse one [C, G, G] parameter and state storage
and read it as different lattices, so varying the geometry varies exactly one
thing. Every internal layout puts the geometry's frequency axis on storage row
b for input band b, which is what makes the one broadcast rule (band b -> row
b, all channels and columns) correct everywhere:

| geometry | frequency axis                          | band b drives                   |
|----------|-----------------------------------------|---------------------------------|
| torus    | lattice rows                            | row b                           |
| cylinder | the open axis (rows)                    | row b                           |
| sheet    | rows (both axes open)                   | row b                           |
| helix    | along the coil, G/4 rows per turn       | ring positions G*b .. G*b+G-1   |
| cube     | the z axis (fully periodic)             | z-slice b (an a x b slab)       |
| sphere   | latitude, south = low -> north          | latitude ring b                 |
"""

from __future__ import annotations

from harness.models.geometries.base import Geometry, PlanarGeometry
from harness.models.geometries.cube import Cube, cube_dims
from harness.models.geometries.cylinder import Cylinder
from harness.models.geometries.helix import Helix
from harness.models.geometries.sheet import Sheet
from harness.models.geometries.sphere import Sphere, sphere_cos_weights, sphere_latitudes
from harness.models.geometries.torus import Torus

#: name -> class, in the order the paper's tables list them
GEOMETRIES: dict[str, type[Geometry]] = {g.name: g for g in (Torus, Cylinder, Sheet, Helix, Cube, Sphere)}
BOUNDARIES = tuple(GEOMETRIES)


def build_geometry(boundary: str, grid: int) -> Geometry:
    """Instantiate one geometry, validating that it can exist at this lattice size."""
    if boundary not in GEOMETRIES:
        raise ValueError(f"unknown boundary '{boundary}': expected one of {BOUNDARIES}")
    return GEOMETRIES[boundary](grid)


__all__ = ["BOUNDARIES", "GEOMETRIES", "Cube", "Cylinder", "Geometry", "Helix", "PlanarGeometry", "Sheet",
           "Sphere", "Torus", "build_geometry", "cube_dims", "sphere_cos_weights", "sphere_latitudes"]

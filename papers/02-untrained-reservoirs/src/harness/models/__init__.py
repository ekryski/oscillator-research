"""The arms under test: the oscillator field, its cores, and its twins."""

from harness.models.baselines import (
    BASELINES,
    CNNBaseline,
    GRUBaseline,
    S4DBaseline,
    TCNBaseline,
    TransformerBaseline,
)
from harness.models.field import (
    OscillatorField,
    physics_block,
    shuffle_kernel_,
    tonotopic_omega,
)
from harness.models.geometries import BOUNDARIES, build_geometry, drive_map
from harness.models.phase import COUPLINGS, PhaseBlock, PhaseCore
from harness.models.random_graph import RandGraphCore
from harness.models.stuart_landau import SLCore

__all__ = ["BASELINES", "BOUNDARIES", "COUPLINGS", "CNNBaseline", "GRUBaseline",
           "OscillatorField", "PhaseBlock", "PhaseCore", "RandGraphCore",
           "S4DBaseline", "SLCore", "TCNBaseline", "TransformerBaseline",
           "build_geometry", "drive_map", "physics_block", "shuffle_kernel_",
           "tonotopic_omega"]

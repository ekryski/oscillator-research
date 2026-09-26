"""Param-matched conventional references, one module per architecture."""

from harness.models.baselines.cnn import CNNBaseline
from harness.models.baselines.gru import GRUBaseline
from harness.models.baselines.pooled import PooledBaseline, register_probe
from harness.models.baselines.s4d import S4DBaseline
from harness.models.baselines.tcn import TCNBaseline
from harness.models.baselines.transformer import TransformerBaseline

#: --arms name -> class, for the runner and the sweep grids
BASELINES = {
    "gru": GRUBaseline,
    "tcn": TCNBaseline,
    "cnn": CNNBaseline,
    "transformer": TransformerBaseline,
    "s4d": S4DBaseline,
}

__all__ = ["BASELINES", "CNNBaseline", "GRUBaseline", "PooledBaseline",
           "S4DBaseline", "TCNBaseline", "TransformerBaseline", "register_probe"]

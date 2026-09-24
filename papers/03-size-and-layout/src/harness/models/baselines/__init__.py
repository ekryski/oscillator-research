"""The trained baselines, one module per architecture."""

from harness.models.baselines.cnn import CNNBaseline
from harness.models.baselines.gru import GRUBaseline
from harness.models.baselines.s4d import S4DBaseline
from harness.models.baselines.tcn import TCNBaseline
from harness.models.baselines.transformer import TransformerBaseline

__all__ = ["CNNBaseline", "GRUBaseline", "S4DBaseline", "TCNBaseline", "TransformerBaseline"]

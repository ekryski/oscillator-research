"""CNN: two causal one-dimensional convolutions."""

from __future__ import annotations

from harness.models.baselines.tcn import TCNBaseline


class CNNBaseline(TCNBaseline):
    """The TCN's form one width notch up, as in paper 02: hidden width 13 (2,109 parameters on 16
    rows) against the TCN's 12. Paper 03 keeps them a notch apart at every budget: the TCN takes the
    widest width at or under the budget, the CNN the narrowest at or over it."""

    def __init__(self, grid: int = 16, hidden: int = 13, k: int = 5):
        super().__init__(grid=grid, hidden=hidden, k=k)

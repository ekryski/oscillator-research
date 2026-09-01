"""How the field is read and how the reading becomes a verdict.

The probe is deliberately the weakest reader in the stack, the floor is the
same probe with the field bypassed, the instruments never decide anything,
and the scorer compares what is left against bars written before the runs.
"""

from harness.measurement.floor import floor_features, floor_features_windowed
from harness.measurement.instruments import (
    analytic_row_phase,
    instrument_field,
    natural_rate,
)
from harness.measurement.probe import (
    collect_features,
    fit_ridge_probe,
    frozen_probe_acc,
    parity_project,
    phase_features,
    pooled_stats,
    train_arm,
    windowed_stats,
)

__all__ = ["analytic_row_phase", "collect_features", "fit_ridge_probe",
           "floor_features", "floor_features_windowed", "frozen_probe_acc",
           "instrument_field", "natural_rate", "parity_project", "phase_features",
           "pooled_stats", "train_arm", "windowed_stats"]

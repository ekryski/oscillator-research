"""Experiment harness for frozen-oscillator-field speech experiments.

Grouped by what each part is FOR, so a reader can find the thing they want to
check without knowing the codebase:

| package        | what lives there                                            |
|----------------|-------------------------------------------------------------|
| `utils`        | the fixed protocol constants; where data and results live    |
| `stimuli`      | sound in — front-ends, stimuli, banks, the injection path    |
| `models`       | the arms under test — the field, its cores, its twins        |
| `measurement`  | how the field is read, and how a reading becomes a verdict   |

Two entry points sit at the top level because they are what you actually run:
`runner` executes one experiment run, and `sweep` plans and drives a whole
matrix across cores.

Inside `models`, geometry gets its own subpackage with one module per lattice
venue, because "which shape" is a factor in the paper and each venue's gluing
rule deserves to be readable on its own.

Importing `harness` gives the flat public surface below. Internal code always
imports the specific submodule instead, which keeps the dependency graph
acyclic and the import cost of a small script small.
"""

from harness.measurement import (
    analytic_row_phase,
    collect_features,
    fit_ridge_probe,
    floor_features,
    frozen_probe_acc,
    instrument_field,
    natural_rate,
    parity_project,
    phase_features,
    pooled_stats,
    train_arm,
    windowed_stats,
)
from harness.models import (
    BASELINES,
    BOUNDARIES,
    COUPLINGS,
    GRUBaseline,
    OscillatorField,
    PhaseCore,
    RandGraphCore,
    SLCore,
    TCNBaseline,
    build_geometry,
    drive_map,
    physics_block,
    shuffle_kernel_,
    tonotopic_omega,
)
from harness.stimuli import (
    band_edges,
    band_index,
    bandpass_rows,
    drive_kick_stats,
    hop_rows,
    hop_rows_quad,
    load_digit_bank,
    make_digit_clips,
    make_digitpair_clips,
    quad_rows_to_drive,
    rows_to_drive,
    tone_classes,
)
from harness.utils import GAIN, PROBE_SCALE, RESULTS_DIR, TWO_PI, WARMUP_FRAMES

__all__ = [
    "BASELINES",
    "BOUNDARIES",
    "COUPLINGS",
    "GAIN",
    "PROBE_SCALE",
    "RESULTS_DIR",
    "TWO_PI",
    "WARMUP_FRAMES",
    "GRUBaseline",
    "OscillatorField",
    "PhaseCore",
    "RandGraphCore",
    "SLCore",
    "TCNBaseline",
    "analytic_row_phase",
    "band_edges",
    "band_index",
    "bandpass_rows",
    "build_geometry",
    "collect_features",
    "drive_kick_stats",
    "drive_map",
    "fit_ridge_probe",
    "floor_features",
    "frozen_probe_acc",
    "hop_rows",
    "hop_rows_quad",
    "instrument_field",
    "load_digit_bank",
    "make_digit_clips",
    "make_digitpair_clips",
    "natural_rate",
    "parity_project",
    "phase_features",
    "physics_block",
    "pooled_stats",
    "quad_rows_to_drive",
    "rows_to_drive",
    "shuffle_kernel_",
    "tone_classes",
    "tonotopic_omega",
    "train_arm",
    "windowed_stats",
]

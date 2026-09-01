"""Stimuli, front-ends, and the fixed injection path into the field."""

from harness.stimuli.digits import (
    DIGIT_BANK_PATH,
    PAIR_GAP_SAMPLES,
    PAIR_LEADER_SAMPLES,
    PAIR_MAX_SAMPLES,
    build_digit_bank,
    load_digit_bank,
    make_digit_clips,
    make_digitpair_clips,
)
from harness.stimuli.filterbank import band_edges, band_index, bandpass_rows
from harness.stimuli.frontend import hop_num_frames, hop_rows, hop_rows_quad
from harness.stimuli.injection import (
    drive_kick_stats,
    quad_rows_to_drive,
    rows_to_drive,
)
from harness.stimuli.synthetic import (
    AM_CARRIER_ROW,
    am_rates,
    make_am_clips,
    make_step_clips,
    make_tone_clips,
    step_freqs,
    tone_classes,
)

__all__ = ["AM_CARRIER_ROW", "DIGIT_BANK_PATH", "PAIR_GAP_SAMPLES",
           "PAIR_LEADER_SAMPLES", "PAIR_MAX_SAMPLES", "am_rates", "band_edges",
           "band_index", "bandpass_rows", "build_digit_bank", "drive_kick_stats",
           "hop_num_frames", "hop_rows", "hop_rows_quad", "load_digit_bank",
           "make_am_clips", "make_digit_clips", "make_digitpair_clips",
           "make_step_clips", "make_tone_clips", "quad_rows_to_drive",
           "rows_to_drive", "step_freqs", "tone_classes"]

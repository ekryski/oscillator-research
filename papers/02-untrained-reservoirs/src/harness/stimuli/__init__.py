"""Sound in: the spoken-digit clips, the fixed front ends, and the injection path into a network."""

from harness.stimuli.digits import (
    DIGIT_MAX_SAMPLES,
    DIGIT_SR,
    DIGIT_TRIM_FRAC,
    PAIR_GAP_SAMPLES,
    PAIR_LEADER_SAMPLES,
    PAIR_MAX_SAMPLES,
    clip_path,
    load_clip,
    read_wav,
)
from harness.stimuli.filterbank import band_edges, band_index, bandpass_rows
from harness.stimuli.frontend import hop_num_frames, hop_rows, hop_rows_quad
from harness.stimuli.injection import drive_kick_stats, quad_rows_to_drive, rows_to_drive

__all__ = ["DIGIT_MAX_SAMPLES", "DIGIT_SR", "DIGIT_TRIM_FRAC", "PAIR_GAP_SAMPLES", "PAIR_LEADER_SAMPLES",
           "PAIR_MAX_SAMPLES", "band_edges", "band_index", "bandpass_rows", "clip_path", "drive_kick_stats",
           "hop_num_frames", "hop_rows", "hop_rows_quad", "load_clip", "quad_rows_to_drive", "read_wav",
           "rows_to_drive"]

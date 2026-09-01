#!/usr/bin/env bash
# Regenerate everything that ships alongside the paper but is not a result:
# the no-dynamics floors, the readout-sufficiency ladder, the trained-head
# baseline protocol, and the audio examples.
#
# Needs the digit bank (see src/data/README.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== the no-dynamics floors"
# one per (representation, noise level): comparing a run to another pathway's
# floor would be comparing it to a different question
for db in 0 5; do
    uv run python -m harness.measurement.floor --frontend mag --noise-db $db --windows 4 --record
    uv run python -m harness.measurement.floor --frontend quad --noise-db $db --windows 4 --record
done
uv run python -m harness.measurement.floor --frontend carrier --noise-db 0 --windows 4 --record
# the order task's matched floor sits at chance by construction; if the
# empirical check disagrees, the task design is broken and nothing else matters
for pair in 3,7 1,8 2,5 4,9 0,6; do
    uv run python -m harness.measurement.floor --task digitpairs --pair $pair \
        --noise-db 0 --windows 4 --record
done

echo "=== the trained-head baseline protocol"
for db in 5 0; do
    uv run python scripts/trained_head_baselines.py --noise-db $db
done

echo "=== the readout-sufficiency ladder"
uv run python scripts/readout_ladder.py

echo "=== audio examples"
uv run python scripts/make_calibration_audio.py
uv run python scripts/make_order_audio.py

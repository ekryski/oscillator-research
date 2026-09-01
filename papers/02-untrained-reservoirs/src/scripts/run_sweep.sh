#!/usr/bin/env bash
# Drive an experiment sweep. A thin wrapper: the grids, the resume guard, the
# resource planning, and the progress reporting all live in harness/sweep.py,
# so there is one place to read what was run rather than two that can disagree.
#
#     bash scripts/run_sweep.sh baselines     # run this first
#     bash scripts/run_sweep.sh envelope      # the primary drive
#     bash scripts/run_sweep.sh quadrature
#     bash scripts/run_sweep.sh carrier
#     bash scripts/run_sweep.sh all           # all four, in order
#
#     bash scripts/run_sweep.sh envelope --dry-run     # what would run
#     bash scripts/run_sweep.sh carrier --workers 2    # override the planner
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec uv run python -m harness.sweep "$@"

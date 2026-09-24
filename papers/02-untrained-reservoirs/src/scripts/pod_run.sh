#!/usr/bin/env bash
# Run the study's tiers on a RunPod pod, or any Linux machine with a GPU.
# One command, top to bottom:
#
#   bash pod_run.sh /workspace/AudioMNIST                 # every tier, in the design's order
#   bash pod_run.sh /workspace/AudioMNIST tier2 becker    # just these
#
# The argument is the AudioMNIST checkout on the attached volume: the folder
# that holds data/01 ... data/60. The script clones or updates the repository,
# builds the bank and the row caches, runs the test suite and the zero-gain
# check, then the tiers. Every step is resume-safe, so re-running it after an
# interruption carries on where it stopped. Results land in
# papers/02-untrained-reservoirs/results/, or under $OSC_RESULTS_DIR when it is
# set (a reproduction run should set it, to keep its record apart); copy that
# folder back (the rsync line is printed at the end).
#
# Pod: one GPU with 24 GB or more, 32 or more vCPUs, 64 GB or more RAM.
set -euo pipefail

AUDIOMNIST="${1:?usage: pod_run.sh <path to the AudioMNIST checkout> [tier ...]}"
shift || true
if [ "$#" -gt 0 ]; then TIERS=("$@"); else TIERS=(gate tier1 tier2 becker tier3 projection carrier); fi
REPO="${REPO:-https://github.com/ekryski/oscillator-research.git}"
BRANCH="${BRANCH:-ek/paper-02}"
WORK="${WORK:-/workspace/oscillator-research}"

[ -d "$AUDIOMNIST/data/01" ] || { echo "no data/01 under $AUDIOMNIST: pass the AudioMNIST checkout" >&2; exit 1; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [ -d "$WORK/.git" ]; then
    git -C "$WORK" fetch origin "$BRANCH" && git -C "$WORK" checkout "$BRANCH" && git -C "$WORK" pull --ff-only
else
    git clone --branch "$BRANCH" "$REPO" "$WORK"
fi
cd "$WORK/papers/02-untrained-reservoirs/src"
uv sync
ln -sfn "$AUDIOMNIST" data/AudioMNIST

CPUS="$(nproc)"
DEVICE="$(uv run python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo "=== $(git rev-parse --short HEAD) on $CPUS cpus, device $DEVICE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# a tier-1 network run holds about 8 GB of features, a tier-2 run about 2 GB; the
# GPU simulates, the CPUs read out, so workers are sized by cores and memory
MEM_GB="$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)"
WIDE=$(( CPUS / 4 < MEM_GB / 10 ? CPUS / 4 : MEM_GB / 10 )); WIDE=$(( WIDE < 1 ? 1 : WIDE ))
NARROW=$(( CPUS / 2 < MEM_GB / 3 ? CPUS / 2 : MEM_GB / 3 )); NARROW=$(( NARROW < 1 ? 1 : NARROW ))

[ -f data/cache/digits_v2.pt ] || uv run python -m harness.experiment.protocol --build-bank
uv run python -m harness.experiment.plan prepare --workers "$(( CPUS < 16 ? CPUS : 16 ))"
uv run pytest -q

for tier in "${TIERS[@]}"; do
    case "$tier" in
        gate|tier1|becker) W="$WIDE"; T=4 ;;
        carrier) W=2; T=4 ;;
        *) W="$NARROW"; T=2 ;;
    esac
    uv run python -m harness.experiment.plan run "$tier" --workers "$W" --threads "$T" --device "$DEVICE"
    if [ "$tier" = gate ]; then uv run python -m harness.experiment.gates check; fi
done
uv run python -m harness.experiment.summary > /dev/null
RESULTS="${OSC_RESULTS_DIR:-$WORK/papers/02-untrained-reservoirs/results}"

echo
echo "=== done. From your machine:"
echo "rsync -avz -e 'ssh -p <PORT> -i ~/.ssh/runpod_ed25519' root@<HOST>:$RESULTS/ <a local folder>/"

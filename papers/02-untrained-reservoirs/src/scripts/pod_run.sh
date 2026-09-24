#!/usr/bin/env bash
# Run the registered confirmatory tiers on a RunPod pod, or any Linux machine
# with a GPU, from the frozen branch. One command, top to bottom:
#
#   bash pod_run.sh /workspace/AudioMNIST                 # every tier, in the registered order
#   bash pod_run.sh /workspace/AudioMNIST tier2 becker    # just these
#
# The argument is the AudioMNIST checkout on the attached volume: the folder
# that holds data/01 ... data/60. The script clones or updates the repository,
# builds both banks and the row caches, runs the test suite and the legacy
# gate, then the tiers. Every step is resume-safe, so re-running it after an
# interruption carries on where it stopped. Results land in
# papers/02-untrained-reservoirs/results/confirmatory/; copy that folder back
# (the rsync line is printed at the end).
#
# Pod: one GPU with 24 GB or more, 32 or more vCPUs, 64 GB or more RAM.
set -euo pipefail

AUDIOMNIST="${1:?usage: pod_run.sh <path to the AudioMNIST checkout> [tier ...]}"
shift || true
if [ "$#" -gt 0 ]; then TIERS=("$@"); else TIERS=(gate tier1 tier2 becker tier3 carrier); fi
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

# a tier-1 field run holds about 8 GB of features, a tier-2 run about 2 GB; the
# GPU simulates, the CPUs read out, so workers are sized by cores and memory
MEM_GB="$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)"
WIDE=$(( CPUS / 4 < MEM_GB / 10 ? CPUS / 4 : MEM_GB / 10 )); WIDE=$(( WIDE < 1 ? 1 : WIDE ))
NARROW=$(( CPUS / 2 < MEM_GB / 3 ? CPUS / 2 : MEM_GB / 3 )); NARROW=$(( NARROW < 1 ? 1 : NARROW ))

uv run python -m harness.stimuli.digits --build-bank      # the exploratory bank, for the legacy gate
[ -f data/cache/digits_v2.pt ] || uv run python -m harness.confirm.protocol --build-bank
uv run python -m harness.confirm.plan prepare --workers "$(( CPUS < 16 ? CPUS : 16 ))"
uv run pytest -q                                          # gate 1
uv run python -m harness.confirm.gates legacy             # gate 2

for tier in "${TIERS[@]}"; do
    case "$tier" in
        gate|tier1|becker) W="$WIDE"; T=4 ;;
        carrier) W=2; T=4 ;;
        *) W="$NARROW"; T=2 ;;
    esac
    uv run python -m harness.confirm.plan run "$tier" --workers "$W" --threads "$T" --device "$DEVICE"
    if [ "$tier" = gate ]; then uv run python -m harness.confirm.gates check; fi
done
uv run python -m harness.confirm.score > ../results/confirmatory/verdicts.txt || true

echo
echo "=== done. From your machine:"
echo "rsync -avz -e 'ssh -p <PORT> -i ~/.ssh/runpod_ed25519' root@<HOST>:$WORK/papers/02-untrained-reservoirs/results/confirmatory/ papers/02-untrained-reservoirs/results/confirmatory/"

#!/usr/bin/env bash
# Run paper 03's tiers on a RunPod pod, or any Linux machine with a GPU, from the pushed branch.
#
#   bash pod_run.sh /workspace/AudioMNIST                               # every tier, in the planned order
#   bash pod_run.sh /workspace/AudioMNIST size trained -- --grids 64 128 # these tiers, these lattices
#
# The first argument is the AudioMNIST checkout on the attached volume (the
# folder that holds data/01 ... data/60). The script clones or updates the
# repository, builds the bank and the row caches, runs the test suite and the
# reuse gate (paper 02's 16 x 16 cells must reproduce here), then the tiers.
# Every step is resume-safe. Results land in
# papers/03-size-and-layout/results/confirmatory/; the rsync line to copy them
# back is printed at the end.
#
# Pod: one GPU with 24 GB or more, 32 or more vCPUs, 64 GB or more RAM. A
# streamed 128 x 128 run holds about 11 GB; the planner prints each run's
# estimate with `plan run <tier> --dry-run`.
set -euo pipefail

AUDIOMNIST="${1:?usage: pod_run.sh <path to the AudioMNIST checkout> [tier ...] [-- plan run options]}"
shift || true
TIERS=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do TIERS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
EXTRA=("$@")
if [ "${#TIERS[@]}" -eq 0 ]; then TIERS=(gate size trained quadrature design carrier design-quadrature design-carrier); fi
REPO="${REPO:-https://github.com/ekryski/oscillator-research.git}"
BRANCH="${BRANCH:-ek/paper-03}"
WORK="${WORK:-/workspace/oscillator-research}"

[ -d "$AUDIOMNIST/data/01" ] || { echo "no data/01 under $AUDIOMNIST: pass the AudioMNIST checkout" >&2; exit 1; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [ -d "$WORK/.git" ]; then
    git -C "$WORK" fetch origin "$BRANCH" && git -C "$WORK" checkout "$BRANCH" && git -C "$WORK" pull --ff-only
else
    git clone --branch "$BRANCH" "$REPO" "$WORK"
fi
cd "$WORK/papers/03-size-and-layout/src"
uv sync
ln -sfn "$AUDIOMNIST" data/AudioMNIST

CPUS="$(nproc)"
DEVICE="$(uv run python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo "=== $(git rev-parse --short HEAD) on $CPUS cpus, device $DEVICE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
MEM_GB="$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)"

[ -f data/cache/digits_v2.pt ] || uv run python -m harness.confirm.protocol --build-bank
uv run python -m harness.confirm.plan prepare --workers "$(( CPUS < 16 ? CPUS : 16 ))"
uv run pytest -q
uv run python -m harness.confirm.gates reuse --device cpu

for tier in "${TIERS[@]}"; do
    case "$tier" in
        carrier|design-carrier) W=2; T=4 ;;                                   # 16,000 steps a clip
        *) W=$(( CPUS / 2 < MEM_GB / 12 ? CPUS / 2 : MEM_GB / 12 )); W=$(( W < 1 ? 1 : W )); T=2 ;;
    esac
    uv run python -m harness.confirm.plan run "$tier" --workers "$W" --threads "$T" --device "$DEVICE" "${EXTRA[@]}"
    if [ "$tier" = gate ]; then uv run python -m harness.confirm.gates check; fi
done
uv run python -m harness.confirm.summary > /dev/null

echo
echo "=== done. From your machine:"
echo "rsync -avz -e 'ssh -p <PORT> -i ~/.ssh/runpod_ed25519' root@<HOST>:$WORK/papers/03-size-and-layout/results/confirmatory/ papers/03-size-and-layout/results/confirmatory/"

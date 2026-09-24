#!/usr/bin/env bash
# Paper 03 on a RunPod pod, or any Linux machine with a GPU, from the pushed branch.
#
#   bash pod_run.sh                                                      # set up, test, benchmark; nothing else
#   bash pod_run.sh /workspace/AudioMNIST gate size -- --grids 8 16 32   # then these tiers, these lattices
#
# First, always: clone or update the repository, sync the environment, run
# the test suite, and run the benchmark (`plan benchmark`), which times one
# batch of every lattice and channel count on the GPU, on the band-energy and
# carrier pathways, with every coupling function and geometry, and
# extrapolates every run and tier. Its report lands in
# papers/03-size-and-layout/results/benchmark/; nothing else runs unless tiers
# are named. BENCHMARK=0 skips it, and BENCHMARK_ARGS passes it options (for
# example "--no-designs").
#
# With tiers: the first argument is the AudioMNIST checkout on the attached
# volume (the folder that holds data/01 ... data/60). The script builds the
# bank and the row caches, runs the reuse check (paper 02's 16 x 16 cells
# re-run on the CPU and compared with its record, read from the clone's
# papers/02-untrained-reservoirs/results/ or from OSC_PAPER02_RESULTS), then
# the tiers. Every step is resume-safe. Results land in
# papers/03-size-and-layout/results/; the rsync lines to copy them back are
# printed at the end.
#
# Pod: one GPU, 32 or more vCPUs, 64 GB or more RAM. 24 GB of GPU memory is
# enough off the carrier pathway; a carrier batch of 32 clips at 1,024 states
# held about 35 GB on the M1 Max, so the carrier wants 48 GB or more (the
# benchmark halves any batch that does not fit and says so). A streamed
# 128 x 128 run holds up to about 23 GB of host memory; `plan run <tier>
# --dry-run` prints each run's estimate.
set -euo pipefail

AUDIOMNIST=""
if [ "$#" -gt 0 ] && [ -d "$1" ]; then AUDIOMNIST="$1"; shift; fi
TIERS=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do TIERS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
EXTRA=("$@")
if [ "${#TIERS[@]}" -gt 0 ] && [ -z "$AUDIOMNIST" ]; then
    echo "usage: pod_run.sh [<path to the AudioMNIST checkout> tier ... [-- plan run options]]" >&2; exit 1
fi
[ -z "$AUDIOMNIST" ] || [ -d "$AUDIOMNIST/data/01" ] || {
    echo "no data/01 under $AUDIOMNIST: pass the AudioMNIST checkout" >&2; exit 1; }
REPO="${REPO:-https://github.com/ekryski/oscillator-research.git}"
BRANCH="${BRANCH:-ek/paper-03}"
WORK="${WORK:-/workspace/oscillator-research}"

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [ -d "$WORK/.git" ]; then
    git -C "$WORK" fetch origin "$BRANCH" && git -C "$WORK" checkout "$BRANCH" && git -C "$WORK" pull --ff-only
else
    git clone --branch "$BRANCH" "$REPO" "$WORK"
fi
cd "$WORK/papers/03-size-and-layout/src"
uv sync
[ -z "$AUDIOMNIST" ] || ln -sfn "$AUDIOMNIST" data/AudioMNIST

CPUS="$(nproc)"
DEVICE="$(uv run python -c 'from harness.utils.device import resolve; print(resolve("auto"))')"   # cuda, mps or cpu
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
echo "=== $(git rev-parse --short HEAD) on $CPUS cpus, device $DEVICE${GPU:+ ($GPU)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
MEM_GB="$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)"

uv run pytest -q

if [ "${BENCHMARK:-1}" != 0 ]; then
    SLUG="$(echo "${GPU:-$DEVICE}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//')"
    OUT="../results/benchmark/benchmark-${SLUG}-$(date +%Y-%m-%d).json"
    # shellcheck disable=SC2086
    uv run python -m harness.experiment.plan benchmark --device "$DEVICE" --out "$OUT" ${BENCHMARK_ARGS:-} \
        | tee "${OUT%.json}.log"
fi

if [ "${#TIERS[@]}" -gt 0 ]; then
    [ -f data/cache/digits_v2.pt ] || uv run python -m harness.experiment.protocol --build-bank
    uv run python -m harness.experiment.plan prepare --workers "$(( CPUS < 16 ? CPUS : 16 ))"
    uv run python -m harness.experiment.gates reuse            # on the CPU, the only device bit-identical to paper 02
    for tier in "${TIERS[@]}"; do
        case "$tier" in
            carrier|design-carrier) W=2; T=4 ;;                               # 16,000 steps a clip
            *) W=$(( CPUS / 2 < MEM_GB / 12 ? CPUS / 2 : MEM_GB / 12 )); W=$(( W < 1 ? 1 : W )); T=2 ;;
        esac
        uv run python -m harness.experiment.plan run "$tier" --workers "$W" --threads "$T" --device "$DEVICE" \
            ${EXTRA[@]+"${EXTRA[@]}"}
        if [ "$tier" = gate ]; then uv run python -m harness.experiment.gates check; fi
    done
    uv run python -m harness.experiment.summary > /dev/null
fi

echo
echo "=== done. From your machine:"
echo "rsync -avz -e 'ssh -p <PORT> -i ~/.ssh/runpod_ed25519' root@<HOST>:$WORK/papers/03-size-and-layout/results/benchmark/ papers/03-size-and-layout/results/benchmark/"
[ "${#TIERS[@]}" -eq 0 ] || echo "rsync -avz -e 'ssh -p <PORT> -i ~/.ssh/runpod_ed25519' root@<HOST>:$WORK/papers/03-size-and-layout/results/ papers/03-size-and-layout/results/"

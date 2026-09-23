#!/usr/bin/env bash
# Start a shell in the lab container with every flag that matters.
#
# Each flag below prevents a distinct failure that is easy to forget:
#   --gpus all       no GPU inside the container without it
#   --shm-size 16g   Docker's default /dev/shm is 64 MB; DataLoader workers
#                    exchange batches through it and die with "bus error"
#   -v repo          code, data, checkpoints and outputs live on the host
#   --user + HOME    outputs owned by you, not root, on a shared machine
#
# Run multi-day jobs from inside tmux ON THE HOST, so an SSH drop does not kill
# the container:   tmux new -s djepa   then   bash docker/run.sh
#
# Usage:  bash docker/run.sh                 # interactive shell
#         bash docker/run.sh <command...>    # run one command and exit
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-demojepa}"

# Untested as non-root at time of writing. If anything fails with a permission
# error inside the container:  RUN_AS_ROOT=1 bash docker/run.sh
if [ "${RUN_AS_ROOT:-0}" = "1" ]; then
    USER_FLAGS=()
else
    USER_FLAGS=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
fi

exec docker run --rm -it \
    --gpus all \
    --shm-size 16g \
    ${USER_FLAGS[@]+"${USER_FLAGS[@]}"} \
    -e WANDB_MODE="${WANDB_MODE:-disabled}" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -v "$REPO":/workspace/Demo-JEPA \
    -w /workspace/Demo-JEPA \
    "$IMAGE" "${@:-bash}"

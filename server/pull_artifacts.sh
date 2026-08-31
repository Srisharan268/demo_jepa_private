#!/usr/bin/env bash
# Continuously PULL logs and checkpoints off a rented GPU box, so a disconnect
# or a reclaimed instance does not lose the run.
#
# Runs on YOUR machine (the 4080 box), not the rented one. Pull, not push:
# vast.ai exposes an SSH host:port, while the 4080 sits behind tailscale and is
# usually not reachable from inside a rented container. Pulling also means
# installing nothing on hardware you are paying for by the hour.
#
# Two cadences, because the artifacts differ by ~4 orders of magnitude:
#
#   logs / CSV   kilobytes   -- every LOG_EVERY seconds
#   *.pt         ~9 GB each  -- every CKPT_EVERY seconds
#
# A stage 1 checkpoint is encoder 4.05GB + dreamer_predictor 1.70GB + Adam
# state 3.40GB. The encoder is FROZEN (unfreeze_vit: False) so those 4.05GB are
# byte-identical every save and already present in vjepa2_ac_repacked.pt --
# which is why checkpoints are pulled on a slow cadence and logs on a fast one.
#
# Metrics do not need this at all: wandb uploads them live. This is for the
# things wandb does not hold -- the CSV and the weights.
#
# Usage:
#   bash server/pull_artifacts.sh root@ssh5.vast.ai 12345
#   bash server/pull_artifacts.sh root@host 12345 ~/djepa_backup 60 900
set -uo pipefail

HOST="${1:?usage: pull_artifacts.sh <user@host> <port> [dest] [log_every_s] [ckpt_every_s]}"
PORT="${2:?missing ssh port}"
DEST="${3:-$HOME/djepa_backup}"
LOG_EVERY="${4:-60}"
CKPT_EVERY="${5:-900}"

REMOTE_REPO="${REMOTE_REPO:-Demo-JEPA}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"

mkdir -p "$DEST"
echo "pulling from $HOST:$PORT/$REMOTE_REPO -> $DEST"
echo "  logs every ${LOG_EVERY}s, checkpoints every ${CKPT_EVERY}s"
echo "  ctrl-c to stop"

if ! $SSH "$HOST" "test -d $REMOTE_REPO"; then
    echo "ERROR: cannot reach $HOST:$PORT or $REMOTE_REPO does not exist there" >&2
    exit 1
fi

pull_logs() {
    # -z helps on text; --partial survives a dropped link mid-file.
    rsync -az --partial -e "$SSH" \
        --include='*/' \
        --include='*.log' --include='*.csv' --include='*.txt' --include='*.json' \
        --exclude='*' \
        "$HOST:$REMOTE_REPO/" "$DEST/" 2>/dev/null
}

pull_ckpts() {
    # No -z: .pt files are dense float data and compress badly, so it just burns
    # CPU. --partial-dir keeps interrupted transfers resumable WITHOUT leaving a
    # truncated file where a valid checkpoint should be -- important, because a
    # half-written .pt that torch.load chokes on is worse than no file.
    rsync -a --partial --partial-dir=.rsync-partial --info=progress2 \
        -e "$SSH" \
        --include='*/' --include='*.pt' --exclude='*' \
        "$HOST:$REMOTE_REPO/exp/" "$DEST/exp/"
}

last_ckpt=0
while true; do
    ts=$(date +%H:%M:%S)
    if pull_logs; then
        echo "[$ts] logs synced"
    else
        echo "[$ts] log sync failed (box down? instance reclaimed?)"
    fi

    now=$(date +%s)
    if [ $((now - last_ckpt)) -ge "$CKPT_EVERY" ]; then
        echo "[$ts] pulling checkpoints (~9GB each, slow)..."
        if pull_ckpts; then
            last_ckpt=$now
            du -sh "$DEST/exp" 2>/dev/null | sed 's/^/          /'
        else
            echo "[$ts] checkpoint sync failed -- will retry next cycle"
        fi
    fi

    sleep "$LOG_EVERY"
done

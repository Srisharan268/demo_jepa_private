#!/usr/bin/env bash
# Stage 2 -- co-training (Dreamer-AC). Requires Stage 1 to have finished:
# meta.dreamer_predictor_checkpoint must point at Stage 1's latest.pt.
#
#   tmux new -s djepa2
#   bash server/run_stage2.sh 2>&1 | tee stage2.log
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-disabled}"

CKPT=$(python - <<'PY'
import yaml
print(yaml.safe_load(open("configs/train/vjepa_2_1_dreamer_ac.yaml"))["meta"]["dreamer_predictor_checkpoint"])
PY
)
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint not found: $CKPT" >&2
    echo "Run stage 1 first, or fix STAGE1_CKPT in server/prepare_configs.py." >&2
    exit 1
fi
echo "stage 1 checkpoint: $CKPT"

exec python -m app.main \
    --fname configs/train/vjepa_2_1_dreamer_ac.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3

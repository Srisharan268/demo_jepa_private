#!/usr/bin/env bash
# Stage 0 -- short in-domain fine-tune of Meta's AC predictor on RLBench franka
# episodes (train split + play). Writes exp/stage0/latest.pt, which stage 2
# then starts from:  python server/prepare_configs.py --stage2-init exp/stage0/latest.pt
#
#   tmux new -s djepa0
#   bash server/run_stage0.sh 2>&1 | tee stage0.log
#   grep "action response" stage0.log | tail      # the go/no-go
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-disabled}"

DATA=$(python - <<'PY'
import yaml
print(yaml.safe_load(open("configs/train/vjepa_2_1_ac.yaml"))["data"]["dataset"])
PY
)
if [ ! -d "$DATA" ] || [ -z "$(find -L "$DATA" -name '*.hdf5' -print -quit)" ]; then
    echo "ERROR: no stage 0 episodes under $DATA -- run server/prepare_configs.py first." >&2
    exit 1
fi
echo "stage 0 data: $DATA"

exec python -m app.main \
    --fname configs/train/vjepa_2_1_ac.yaml \
    --devices cuda:0

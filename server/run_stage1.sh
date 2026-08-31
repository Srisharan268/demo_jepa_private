#!/usr/bin/env bash
# Stage 1 -- Dreamer Predictor. Run under tmux (or via sbatch); this is a
# multi-hour job and dies with its shell otherwise.
#
#   tmux new -s djepa
#   bash server/run_stage1.sh 2>&1 | tee stage1.log
#   <ctrl-b d to detach>
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# `python -m` puts cwd on sys.path but `python file.py` does not -- set it
# explicitly so `import app` resolves however this is invoked.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Reduces allocator fragmentation; matters because activations swing a lot
# between the encoder's no_grad pass and the Dreamer backward.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-disabled}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv

exec python -m app.main \
    --fname configs/train/vjepa_2_1_dreamer_predictor.yaml \
    --devices cuda:0

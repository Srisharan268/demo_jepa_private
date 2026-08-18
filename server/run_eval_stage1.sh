#!/usr/bin/env bash
# Stage 1 evaluation -- cross-embodiment retrieval accuracy.
#
# Uses the paper's own eval, app/vjepa_2_1_dreamer_predictor/retrieval_eval.py.
# Given a Dreamer prediction it scores whether the matching target feature is
# the nearest neighbour among the batch (acc@1, acc@k). Chance = 1/batch_size,
# so report the batch size alongside the number.
#
# Usage:
#   bash server/run_eval_stage1.sh data/val [checkpoint]
#
# Defaults to the stage 1 config's own dreamer_predictor output if no
# checkpoint is given.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VAL_DATA="${1:-data/val}"
CKPT="${2:-}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# retrieval_eval.py calls wandb.init() unconditionally -- disable it or the run
# blocks on an auth prompt.
export WANDB_MODE="${WANDB_MODE:-disabled}"

EVAL_CFG="configs/train/_eval_stage1.yaml"

python - "$VAL_DATA" "$CKPT" "$EVAL_CFG" <<'PY'
import sys, yaml
val, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
c = yaml.safe_load(open("configs/train/vjepa_2_1_dreamer_predictor.yaml"))
c["data"]["dataset"] = val
# Eval only -- no accumulation, and a larger batch makes acc@1 a harder,
# more informative test (chance = 1/batch_size).
c["data"]["batch_size"] = 16
c["optimization"]["accum_steps"] = 1
if ckpt:
    c["meta"]["dreamer_predictor_checkpoint"] = ckpt
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
print(f"eval config  -> {out}")
print(f"  dataset    : {val}")
print(f"  dreamer    : {c['meta']['dreamer_predictor_checkpoint']}")
print(f"  batch/chance: {c['data']['batch_size']} / {1/c['data']['batch_size']:.3f}")
PY

exec python -m app.vjepa_2_1_dreamer_predictor.retrieval_eval \
    --fname "$EVAL_CFG" --devices cuda:0 --debugmode True

#!/usr/bin/env python3
"""Write the Stage 1 / Stage 2 training configs for a 4x32GB node.

Starts from the upstream YAMLs and changes ONLY what is justified:

  paths        folder / dataset / checkpoints  (environment, not method)
  camera_views right_shoulder_rgb -- RLBench writes this; the yaml's
               `camera_front` does not exist in collected data
  data_type    "sim" -- selects the franka/sawyer preset; "real" looks
               for a ur/ directory
  batch_size   sized so global batch matches the paper on 4 GPUs
  accum_steps  Stage 1 only, to hit global batch 128 without OOM

Everything else -- model, crop_size, dataset_fpcs, epochs, ipe, lr, warmup,
anneal, weight decay, losses -- is left exactly as upstream.

Global batch arithmetic (paper ran 8 GPUs; we have 4):

  Stage 1   paper 16 x 8 = 128   ours 8 x 4 x accum 4 = 128
  Stage 2   paper  2 x 8 =  16   ours 4 x 4            =  16

Stage 1 cannot simply double to 32/GPU: measured activation cost is
~1.0-1.9 GB per sample (DreamerPredictor's Conv3dFusionNetwork is not
activation-checkpointed), so batch 16 alone needs ~34-47 GB. Stage 2 keeps
dreamer_predictor under no_grad, so it has room to double instead and needs
no code change at all.

Usage:  python server/prepare_configs.py
"""
import os
import sys

import yaml

# ----------------------------------------------------------------------------
# EDIT THESE
# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET = "/path/to/rlbench/pairs"              # collected RLBench pairs
STAGE0_CKPT = "/path/to/vjepa2_ac_repacked.pt"  # Meta V-JEPA 2.1-AC, repacked
OUT_STAGE1 = "/path/to/exp/stage1"
OUT_STAGE2 = "/path/to/exp/stage2"

CAMERA = "right_shoulder_rgb"
NUM_WORKERS = 8          # per rank; 4 ranks => 32 loader processes. Lower if host RAM is tight.
N_GPUS = 4

# Stage 1's output, consumed by Stage 2. Written by the Stage 1 run.
STAGE1_CKPT = os.path.join(OUT_STAGE1, "latest.pt")
# ----------------------------------------------------------------------------

S1 = "configs/train/vjepa_2_1_dreamer_predictor.yaml"
S2 = "configs/train/vjepa_2_1_dreamer_ac.yaml"


def load(rel):
    with open(os.path.join(REPO, rel)) as f:
        return yaml.safe_load(f)


def save(rel, cfg):
    with open(os.path.join(REPO, rel), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def expect(cfg, path, want, rel):
    """Fail loudly if upstream changed under us."""
    node = cfg
    for k in path[:-1]:
        node = node[k]
    got = node.get(path[-1])
    if got != want:
        sys.exit(f"ERROR: {rel}: expected {'.'.join(path)}={want!r}, found {got!r}.\n"
                 f"Upstream changed -- re-check the memory analysis before proceeding.")


def report(tag, cfg, accum):
    d = cfg["data"]
    g = d["batch_size"] * N_GPUS * accum
    print(f"  {tag}: batch_size={d['batch_size']} x {N_GPUS} GPUs x accum {accum} "
          f"= global {g}")
    for key, val in (("dataset", d["dataset"]),
                     ("pretrain_checkpoint", cfg["meta"]["pretrain_checkpoint"])):
        mark = "ok " if val and os.path.exists(str(val)) else "MISSING"
        print(f"      {key:22s} {val}  [{mark}]")


# ---------------------------------------------------------------- Stage 1 ---
c = load(S1)
expect(c, ["data", "batch_size"], 16, S1)
expect(c, ["optimization", "epochs"], 315, S1)
expect(c, ["optimization", "ipe"], 300, S1)

c["folder"] = OUT_STAGE1
c["data"]["dataset"] = DATASET
c["data"]["camera_views"] = [CAMERA]
c["data"]["data_type"] = "sim"
c["data"]["batch_size"] = 8
c["data"]["num_workers"] = NUM_WORKERS
c["optimization"]["accum_steps"] = 4
c["meta"]["pretrain_checkpoint"] = STAGE0_CKPT
c["meta"]["dreamer_predictor_checkpoint"] = None
save(S1, c)

# ---------------------------------------------------------------- Stage 2 ---
d = load(S2)
expect(d, ["data", "batch_size"], 2, S2)
expect(d, ["optimization", "epochs"], 315, S2)
expect(d, ["optimization", "ipe"], 300, S2)

d["folder"] = OUT_STAGE2
d["data"]["dataset"] = DATASET
d["data"]["camera_views"] = [CAMERA]
d["data"]["data_type"] = "sim"
d["data"]["batch_size"] = 4          # 4 x 4 GPUs = 16 = paper's global batch
d["data"]["num_workers"] = NUM_WORKERS
# Stage 2 needs no accumulation, so app/vjepa_2_1_dreamer_ac/train.py is untouched.
d["meta"]["pretrain_checkpoint"] = STAGE0_CKPT
d["meta"]["dreamer_predictor_checkpoint"] = STAGE1_CKPT
d["meta"]["load_predictor"] = True
save(S2, d)

print("configs written\n")
report("stage 1", load(S1), 4)
report("stage 2", load(S2), 1)
print("\nUnchanged from upstream: model, crop_size, dataset_fpcs, epochs, ipe,")
print("lr, start_lr, final_lr, warmup, anneal, weight_decay, loss settings.")

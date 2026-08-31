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
import argparse
import os
import subprocess
import sys

import yaml

_ap = argparse.ArgumentParser(description="Write stage 1/2 training configs.")
_ap.add_argument("--gpus", type=int, default=4, help="GPUs in the run (world size)")
_ap.add_argument("--epochs", type=int, default=None, help="override epochs for both stages")
# WSD's total length is epochs * ipe, so lowering ipe and raising epochs by the
# same factor is the SAME schedule -- but CHECKPOINT_FREQ=1 saves latest.pt every
# epoch, so it also checkpoints proportionally more often. That is the cheap way
# to bound how much a crash costs: `--epochs 12 --ipe 100` == `--epochs 4 --ipe
# 300` for the optimiser, and gives 12 checkpoints instead of 4.
# warmup/anneal are ABSOLUTE (in epochs), not fractions -- scale them yourself.
_ap.add_argument("--ipe", type=int, default=None, help="iterations per epoch (checkpoint granularity)")
# latest.pt is OVERWRITTEN every epoch (CHECKPOINT_FREQ=1). Only e{N}.pt is
# kept, and only when epoch % save_every_freq == 0. Upstream ships 25, so a
# short run keeps ONLY e0.pt -- an untrained model -- and nothing else.
# Each checkpoint is ~9GB, so set this against available disk, not habit.
_ap.add_argument("--save-every", type=int, default=None,
                 help="keep e{N}.pt every N epochs (~9GB each); upstream 25")
_ap.add_argument("--warmup", type=int, default=None, help="warmup epochs (absolute)")
_ap.add_argument("--anneal", type=int, default=None, help="anneal epochs (absolute)")
_ap.add_argument("--smoke", action="store_true",
                 help="tiny plumbing run: batch 1, accum 1, epochs 1, ipe 20")
ARGS = _ap.parse_args()

# ----------------------------------------------------------------------------
# EDIT THESE
# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Repo-relative, so the same config works on laptop and server with no edits.
# Produced by: python server/split_dataset.py
DATASET = os.path.join(REPO, "data", "train")
HELD_OUT = os.path.join(REPO, "data", "val")     # used by eval + deploy, not training
OUT_STAGE1 = os.path.join(REPO, "exp", "stage1")  # exp/ is gitignored
OUT_STAGE2 = os.path.join(REPO, "exp", "stage2")

# Not in the repo -- downloaded on the server (see RUNBOOK step 6).
STAGE0_CKPT = os.path.expanduser("~/vjepa2_ac_repacked.pt")

CAMERA = "right_shoulder_rgb"
NUM_WORKERS = 8          # per rank; 4 ranks => 32 loader processes. Lower if host RAM is tight.

# Number of GPUs actually available. Override with --gpus N.
# NOTE: this drives the arithmetic here AND the --devices list the run scripts
# use (via DJEPA_DEVICES, printed at the end). Changing it alone is not enough
# if you launch the scripts by hand.
N_GPUS = int(os.environ.get("DJEPA_GPUS", "4"))
for _i, _a in enumerate(sys.argv):
    if _a == "--gpus" and _i + 1 < len(sys.argv):
        N_GPUS = int(sys.argv[_i + 1])

# Paper global batches, preserved wherever possible.
PAPER_GLOBAL_S1 = 128    # 16 x 8 GPUs
PAPER_GLOBAL_S2 = 16     # 2 x 8 GPUs

# Stage 1's output, consumed by Stage 2. Written by the Stage 1 run.
STAGE1_CKPT = os.path.join(OUT_STAGE1, "latest.pt")
# ----------------------------------------------------------------------------

FATAL = []

S1 = "configs/train/vjepa_2_1_dreamer_predictor.yaml"
S2 = "configs/train/vjepa_2_1_dreamer_ac.yaml"


def load(rel):
    """Load the UPSTREAM config from git HEAD, not the working tree.

    This script overwrites the same files it reads, so reading the working
    tree makes it non-idempotent: a second run sees the first run's output
    and the expect() guards fire against our own values. Reading HEAD means
    every run starts from the pristine upstream config, no matter how many
    times it has run before.
    """
    blob = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        cwd=REPO, capture_output=True, text=True,
    )
    if blob.returncode != 0:
        sys.exit(f"ERROR: cannot read {rel} from git HEAD:\n{blob.stderr.strip()}\n"
                 f"Are you inside the repo, and is {rel} committed?")
    return yaml.safe_load(blob.stdout)


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


def count_pairs(root):
    """Dataset __len__ is the number of paired EPISODES, not frames."""
    if not os.path.isdir(root):
        return None
    n = 0
    for task in os.listdir(root):
        d = os.path.join(root, task, "franka")
        if os.path.isdir(d):
            n += len([f for f in os.listdir(d) if f.endswith((".hdf5", ".h5"))])
    return n


def report(tag, cfg, accum):
    d = cfg["data"]
    g = d["batch_size"] * N_GPUS * accum
    print(f"  {tag}: batch_size={d['batch_size']} x {N_GPUS} GPUs x accum {accum} "
          f"= global {g}")
    for key, val in (("dataset", d["dataset"]),
                     ("pretrain_checkpoint", cfg["meta"]["pretrain_checkpoint"])):
        mark = "ok " if val and os.path.exists(str(val)) else "MISSING"
        print(f"      {key:22s} {val}  [{mark}]")

    # The dataloader uses drop_last=True and the dataset yields ONE sample per
    # episode pair. If batch_size exceeds what each rank receives, len(loader)
    # is 0 and training loops forever on StopIteration -- a silent hang, not an
    # error. Catch it here instead.
    pairs = count_pairs(d["dataset"])
    if pairs is None:
        return
    per_rank = pairs // N_GPUS
    batches = per_rank // d["batch_size"]
    print(f"      {'episode pairs':22s} {pairs}  ->  {per_rank}/rank  ->  "
          f"{batches} batches/rank/epoch")
    if batches == 0:
        max_bs = max(per_rank, 1)
        print(f"      *** FATAL: batch_size {d['batch_size']} > {per_rank} samples per rank.")
        print(f"          len(loader)==0 with drop_last=True; training will HANG silently.")
        print(f"          Need >= {d['batch_size'] * N_GPUS} episode pairs, or batch_size <= {max_bs}.")
        FATAL.append(tag)



def apply_schedule_overrides(cfg):
    """Apply --ipe/--warmup/--anneal, and sanity-check the WSD phases."""
    o = cfg["optimization"]
    if ARGS.ipe:
        o["ipe"] = ARGS.ipe
    if ARGS.save_every is not None:
        cfg["meta"]["save_every_freq"] = ARGS.save_every
        kept = len(range(0, o["epochs"], ARGS.save_every)) if ARGS.save_every > 0 else 0
        print(f"  keeping {kept} e{{N}}.pt checkpoint(s) + latest.pt "
              f"~= {(kept + 1) * 9} GB")
    for key in ("warmup", "anneal"):
        val = getattr(ARGS, key)
        if val is not None:
            o[key] = val
    # WSDSchedule sets T_max = total - warmup - anneal (schedulers.py:19). If the
    # phases exceed the run, the flat phase vanishes and the LR curve is not what
    # you think it is.
    total = o["epochs"]
    if o.get("warmup", 0) + o.get("anneal", 0) >= total:
        sys.exit(f"ERROR: warmup {o.get('warmup')} + anneal {o.get('anneal')} >= "
                 f"epochs {total}. No flat phase would remain -- WSD degenerates.")
    return cfg


# ---------------------------------------------------------------- Stage 1 ---
c = load(S1)
expect(c, ["data", "batch_size"], 16, S1)
expect(c, ["optimization", "epochs"], 315, S1)
expect(c, ["optimization", "ipe"], 300, S1)

c["folder"] = OUT_STAGE1
c["data"]["dataset"] = DATASET
c["data"]["camera_views"] = [CAMERA]
c["data"]["data_type"] = "sim"
# Held-out split -> in-loop validation (train.py). Without it you cannot tell
# learning from memorising on a small dataset.
c["data"]["val_dataset"] = HELD_OUT
# batch_size 8 is the memory ceiling per GPU (measured ~1.0-1.9 GB/sample of
# activations). Accumulation makes up the rest of the paper's global batch.
S1_BATCH = 1 if ARGS.smoke else 8
# Smoke uses accum 2, not 1: with n_micro == 1 the accumulation loop runs once,
# no_sync() is never entered and the /n_micro scaling is a no-op -- the whole
# custom code path would go untested. 2 exercises it for ~nothing.
S1_ACCUM = 2 if ARGS.smoke else max(1, PAPER_GLOBAL_S1 // (S1_BATCH * N_GPUS))
c["data"]["batch_size"] = S1_BATCH
c["data"]["num_workers"] = 2 if ARGS.smoke else NUM_WORKERS
c["optimization"]["accum_steps"] = S1_ACCUM
if ARGS.smoke:
    # Plumbing only: 20 optimizer steps, no LR schedule to speak of.
    c["optimization"].update(epochs=1, ipe=20, warmup=0, anneal=0)
else:
    if ARGS.epochs:
        c["optimization"]["epochs"] = ARGS.epochs
    apply_schedule_overrides(c)
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
# Stage 2 has NO accumulation support (app/vjepa_2_1_dreamer_ac/train.py is
# untouched), so global batch is batch_size x N_GPUS. On few GPUs the batch
# needed to hit 16 may not fit; cap it and report the shortfall honestly.
S2_BATCH_IDEAL = max(1, PAPER_GLOBAL_S2 // N_GPUS)
S2_BATCH_CAP = 4                      # per-GPU memory ceiling for stage 2
S2_BATCH = 1 if ARGS.smoke else min(S2_BATCH_IDEAL, S2_BATCH_CAP)
d["data"]["batch_size"] = S2_BATCH
d["data"]["num_workers"] = 2 if ARGS.smoke else NUM_WORKERS
if ARGS.smoke:
    d["optimization"].update(epochs=1, ipe=20, warmup=0, anneal=0)
else:
    if ARGS.epochs:
        d["optimization"]["epochs"] = ARGS.epochs
    apply_schedule_overrides(d)
# Stage 2 needs no accumulation, so app/vjepa_2_1_dreamer_ac/train.py is untouched.
d["meta"]["pretrain_checkpoint"] = STAGE0_CKPT
d["meta"]["dreamer_predictor_checkpoint"] = STAGE1_CKPT
d["meta"]["load_predictor"] = True
save(S2, d)

print("configs written\n")
# Report from the in-memory configs we just wrote -- load() now reads git HEAD,
# so it would report upstream's values, not ours.
report("stage 1", c, S1_ACCUM)
report("stage 2", d, 1)
print("\nUnchanged from upstream: model, crop_size, dataset_fpcs, epochs, ipe,")
print("lr, start_lr, final_lr, warmup, anneal, weight_decay, loss settings.")

if FATAL:
    print("\n" + "=" * 72)
    print(f"REFUSING TO PROCEED: {', '.join(FATAL)} would hang (see FATAL above).")
    print("The configs were written, but do NOT launch training with them.")
    print("")
    print("Fix by either:")
    print("  (a) collecting more episodes -- the real fix; or")
    print("  (b) for a pipeline smoke test only, lowering batch_size in the")
    print("      config(s) to at most the per-rank sample count shown above.")
    print("      Note this also means memory is NOT validated at the batch size")
    print("      you will eventually train with.")
    print("=" * 72)
    sys.exit(1)

#!/usr/bin/env python3
"""Prove that skipping DDP at world_size==1 changes nothing numerically.

Runs stage 1 twice with an IDENTICAL config -- once with the NoDDP shim, once
forced back onto real DistributedDataParallel -- and compares the per-iteration
losses from exp/stage1/log_r0.csv bit-for-bit.

Stage 1 is deterministic: two runs of the same config previously reproduced
0.971 / 0.604 / 0.540 exactly (HANDOFF §5b). So identical losses is PROOF of
equivalence, not an argument for it. Any difference means the reasoning about
DDP being inert at one rank is wrong somewhere, and the change must be reverted.

The DDP path is forced by DJEPA_FORCE_DDP=1, which wrap_ddp() honours.

Usage:
  python server/verify_noddp.py                 # 20 steps, batch 8
  python server/verify_noddp.py --steps 10 --batch 4
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = "configs/train/vjepa_2_1_dreamer_predictor.yaml"
CSV_PATH = os.path.join(REPO, "exp", "stage1", "log_r0.csv")


def run(tag, force_ddp, steps, env_base):
    # Remove the CSV so we read only this run's rows -- CSVLogger appends.
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    env = dict(env_base)
    if force_ddp:
        env["DJEPA_FORCE_DDP"] = "1"
    else:
        env.pop("DJEPA_FORCE_DDP", None)

    log = os.path.join(REPO, f"verify_{tag}.log")
    print(f"\n=== {tag} (force_ddp={force_ddp}, {steps} steps) ===", flush=True)
    with open(log, "w") as f:
        rc = subprocess.call(
            [sys.executable, "-m", "app.main", "--fname", CFG,
             "--devices", "cuda:0", "--debugmode", "True"],
            cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
        )
    if rc != 0:
        tail = "\n".join(open(log, errors="replace").read().splitlines()[-15:])
        sys.exit(f"ERROR: {tag} exited {rc}. See {log}\n{tail}")

    with open(CSV_PATH) as f:
        rows = [r for r in csv.DictReader(f)]
    losses = [r["loss"].strip() for r in rows]          # strings: exact compare
    peak = max(float(r.get("gpu-time(ms)", 0) or 0) for r in rows) if rows else 0
    print(f"  {len(losses)} iterations, first={losses[0]} last={losses[-1]}")
    shutil.copy(CSV_PATH, os.path.join(REPO, f"verify_{tag}.csv"))
    return losses, peak, log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = "disabled"          # a control run is not an experiment

    # Identical config for both. --measure pins epochs 1 / accum 1 / no schedule;
    # ipe is then set to --steps. Seed is fixed in the config (239).
    subprocess.check_call(
        [sys.executable, os.path.join(REPO, "server", "set_batch.py"),
         "--batch", str(args.batch), "--measure"], cwd=REPO)
    import yaml
    c = yaml.safe_load(open(os.path.join(REPO, CFG)))
    c["optimization"]["ipe"] = args.steps
    # Validation would add forward passes between the runs; keep the comparison
    # to the training path alone.
    c["data"].pop("val_dataset", None)
    yaml.safe_dump(c, open(os.path.join(REPO, CFG), "w"), sort_keys=False)

    a, peak_a, log_a = run("noddp", False, args.steps, env)
    b, peak_b, log_b = run("ddp", True, args.steps, env)

    print("\n" + "=" * 62)
    if a == b:
        print(f"PASS -- all {len(a)} per-iteration losses BIT-IDENTICAL.")
        print("Skipping DDP at world_size==1 is numerically equivalent.")
    else:
        print("*** FAIL -- losses differ. DO NOT keep the NoDDP change. ***")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"  first divergence at iter {i}: noddp={x}  ddp={y}")
                break
        if len(a) != len(b):
            print(f"  iteration counts differ: {len(a)} vs {len(b)}")
    print("=" * 62)
    print(f"logs: {log_a}  {log_b}")
    print("Peak GPU memory per run is in those logs as [mem: ...] -- the whole")
    print("point of the change is that NoDDP's is several GB lower.")
    return 0 if a == b else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Split the paired RLBench dataset into train / held-out roots.

The dataloader expects a root containing <task>/<robot>/*.hdf5, so this
produces two such roots rather than splitting in place.

Episodes are paired across embodiments (same filename in franka/ and sawyer/),
so the split is done on filenames and applied to both -- splitting them
independently would break the pairing the method depends on.

Deterministic (seed 0), matching the split used during Colab testing.

Usage:
  python server/split_dataset.py                       # defaults below
  python server/split_dataset.py --val 6 --seed 0
"""
import argparse
import os
import random
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SRC = os.path.join(REPO, "data", "rlbench_data")
DEFAULT_TRAIN = os.path.join(REPO, "data", "train")
DEFAULT_VAL = os.path.join(REPO, "data", "val")
ROBOTS = ("franka", "sawyer")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--val", type=int, default=6, help="held-out episodes per task")
    p.add_argument("--val-dir", default=DEFAULT_VAL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--move", action="store_true", help="move instead of copy (saves disk)")
    args = p.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f"ERROR: source not found: {args.src}")

    tasks = sorted(d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d)))
    if not tasks:
        sys.exit(f"ERROR: no task directories under {args.src}")

    for root in (args.train, args.val_dir):
        shutil.rmtree(root, ignore_errors=True)

    xfer = shutil.move if args.move else shutil.copy
    total_tr = total_va = 0

    for task in tasks:
        per_robot = {}
        for robot in ROBOTS:
            d = os.path.join(args.src, task, robot)
            if not os.path.isdir(d):
                sys.exit(f"ERROR: missing {d} -- expected subdirs {ROBOTS}")
            per_robot[robot] = sorted(f for f in os.listdir(d) if f.endswith((".hdf5", ".h5")))

        a, b = per_robot[ROBOTS[0]], per_robot[ROBOTS[1]]
        if a != b:
            only_a, only_b = set(a) - set(b), set(b) - set(a)
            sys.exit(f"ERROR: {task}: episode pairing broken.\n"
                     f"  only in {ROBOTS[0]}: {sorted(only_a)[:5]}\n"
                     f"  only in {ROBOTS[1]}: {sorted(only_b)[:5]}")

        names = list(a)
        if args.val >= len(names):
            sys.exit(f"ERROR: {task}: --val {args.val} but only {len(names)} episodes")

        random.Random(args.seed).shuffle(names)
        val_names, train_names = names[: args.val], names[args.val:]

        for root, subset in ((args.train, train_names), (args.val_dir, val_names)):
            for robot in ROBOTS:
                dst = os.path.join(root, task, robot)
                os.makedirs(dst, exist_ok=True)
                for n in subset:
                    xfer(os.path.join(args.src, task, robot, n), os.path.join(dst, n))

        total_tr += len(train_names)
        total_va += len(val_names)
        print(f"  {task:24s} train {len(train_names):4d}   held-out {len(val_names):4d}")

    print(f"\ntrain    -> {args.train}   ({total_tr} episode pairs)")
    print(f"held-out -> {args.val_dir}   ({total_va} episode pairs)")
    print(f"\nSet these in server/prepare_configs.py (DATASET) and use the held-out")
    print(f"root for run_eval_stage1.sh and the deploy reference demo.")

    if total_tr < 100:
        print(f"\nNOTE: {total_tr} training pairs is a smoke-test dataset, not a training set.")
        print(f"The configs run 94,500 optimizer steps at global batch 128 -- thousands of")
        print(f"passes over this data. Fine for validating the pipeline; collect more")
        print(f"(README uses --total_episodes 200) before any run you intend to report.")


if __name__ == "__main__":
    main()

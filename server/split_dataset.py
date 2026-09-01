#!/usr/bin/env python3
"""Split the paired RLBench dataset into train / held-out roots.

The dataloader expects a root containing <task>/<robot>/*.hdf5 with
<robot> being exactly `franka` and `sawyer` for data_type: sim -- see
app/vjepa_2_1_dreamer_predictor/dataset.py:27 and
app/vjepa_2_1_dreamer_ac/dataset.py:29, which hardcode primary_subdir="franka".
RLBench's own API calls that arm `panda`, so a collector writing `panda/` must
be renamed to `franka/` or nothing will load.

Two modes:
  IN-PLACE (default, --src == --train): moves the held-out episodes out of
    train/ into val/. Deletes nothing.
  COPY (--src elsewhere): rebuilds train/ and val/ from --src, and refuses if
    either already holds data.

Episodes are paired across embodiments (same filename in franka/ and sawyer/),
so the split is done on filenames and applied to both -- splitting them
independently would break the pairing the method depends on.

Deterministic (seed 0), matching the split used during Colab testing.

Usage:
  python server/split_dataset.py --val 40         # in-place: train/ -> val/
  python server/split_dataset.py --src data/collected --val 40   # copy mode
"""
import argparse
import os
import random
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default to data/train: collection writes there, and this script now splits
# IN PLACE when --src equals --train (moves the held-out episodes to data/val,
# deletes nothing). The old default pointed at the small in-repo smoke extract,
# which meant the default invocation WIPED collected data and replaced it with
# 18 episodes. That happened. Do not restore it.
DEFAULT_SRC = os.path.join(REPO, "data", "train")
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
    p.add_argument("--force", action="store_true",
                   help="allow deleting a destination that already holds data")
    args = p.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f"ERROR: source not found: {args.src}\n"
                 "Pass --src pointing at your COLLECTED episodes. The default\n"
                 "is only the small in-repo smoke extract.")

    tasks = sorted(d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d)))
    if not tasks:
        sys.exit(f"ERROR: no task directories under {args.src}")

    # This used to rmtree train/ and val/ unconditionally, then repopulate from
    # --src. If --src pointed anywhere other than where the real data lived,
    # that DESTROYED the dataset -- and it did: 402 freshly collected pairs were
    # wiped and replaced by the 18-episode in-repo smoke extract, because --src
    # defaults to data/rlbench_data.
    # Refuse to delete a non-empty destination unless told explicitly.
    def _count(root):
        n = 0
        for _dp, _dn, files in os.walk(root):
            n += sum(1 for f in files if f.endswith((".hdf5", ".h5")))
        return n

    src_n = _count(args.src)
    same = os.path.abspath(args.src) == os.path.abspath(args.train)

    if same:
        # IN-PLACE: data was collected straight into data/train, which is the
        # normal workflow. Carve the held-out set out by MOVING episodes to
        # data/val. Nothing is ever deleted, so a mistake costs a re-split, not
        # the dataset.
        if _count(args.val_dir) and not args.force:
            sys.exit(f"ERROR: {args.val_dir} already holds "
                     f"{_count(args.val_dir)} episode file(s).\n"
                     "Move them back into train/ first, or pass --force.")
        print(f"in-place split: holding out {args.val}/task from {args.src}\n")
    else:
        for root in (args.train, args.val_dir):
            have = _count(root)
            if have and not args.force:
                sys.exit(
                    "ERROR: refusing to delete existing data.\n"
                    f"  {root}\n"
                    f"    holds {have} episode file(s) and would be DELETED.\n"
                    f"  --src {args.src}\n"
                    f"    holds {src_n}.\n"
                    "\n"
                    "  If the destination is where your collected data actually is,\n"
                    "  this would destroy it. Point --src at it, or move it to safety.\n"
                    "  Re-run with --force once you are certain."
                )
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

        if same:
            # Only the held-out episodes move; the rest already sit in train/.
            for robot in ROBOTS:
                dst = os.path.join(args.val_dir, task, robot)
                os.makedirs(dst, exist_ok=True)
                for n in val_names:
                    shutil.move(os.path.join(args.src, task, robot, n),
                                os.path.join(dst, n))
        else:
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

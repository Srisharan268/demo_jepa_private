#!/usr/bin/env python3
"""Set stage 1's per-GPU batch size, holding the paper's global batch constant.

    global batch = batch_size x world_size x accum_steps

The paper ran 16 x 8 GPUs = 128, and lr / warmup / the WSD schedule are all
tuned for that. So batch_size and accum_steps must move in opposite directions:
changing the GLOBAL batch changes the optimisation problem, not just the speed.

    batch  8 x accum 16 = 128   (fits 32GB -- the lab card)
    batch 32 x accum  4 = 128   (fits 96GB -- rented)

Both do the SAME 128 sample-forwards per optimizer step and the same FLOPs.
Larger micro-batches are faster only through lower overhead -- fewer Python
iterations, fewer kernel launches, better SM occupancy. Expect ~1.2-1.6x, NOT
the 4x the accum ratio suggests. Verify by measurement; do not assume.

Run this AFTER prepare_configs.py. That script reads its baseline from
`git show HEAD:` and rewrites the file, so it would undo this edit. This script
deliberately reads the WORKING TREE instead -- it is an override layered on top
of a freshly written config, not a generator.

Usage:
  python server/set_batch.py --batch 32
  python server/set_batch.py --batch 8 --measure      # short run for timing
  python server/set_batch.py --batch 32 --stage 2
"""
import argparse
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGS = {
    1: "configs/train/vjepa_2_1_dreamer_predictor.yaml",
    2: "configs/train/vjepa_2_1_dreamer_ac.yaml",
}
PAPER_GLOBAL = {1: 128, 2: 16}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, required=True, help="per-GPU micro-batch")
    p.add_argument("--stage", type=int, default=1, choices=(1, 2))
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--global-batch", type=int, default=None,
                   help="override the paper's global batch (rarely correct)")
    p.add_argument("--measure", action="store_true",
                   help="epochs 1, ipe 50, warmup/anneal 0 -- for throughput only")
    args = p.parse_args()

    rel = CFGS[args.stage]
    path = os.path.join(REPO, rel)
    c = yaml.safe_load(open(path))

    target = args.global_batch or PAPER_GLOBAL[args.stage]
    per_step = args.batch * args.gpus

    if args.measure:
        # Timing only: accum is forced to 1 below, so the global batch is
        # irrelevant and divisibility into 128 must not block the measurement.
        # This is how you measure batch sizes like 48 that do not divide 128.
        accum = 1
    elif args.stage == 2:
        # app/vjepa_2_1_dreamer_ac/train.py is untouched upstream and has NO
        # accumulation support, so global batch is batch x world_size, full stop.
        accum = 1
        if per_step != target:
            print(f"NOTE: stage 2 has no accum_steps; global batch is {per_step}, "
                  f"not the paper's {target}.")
    else:
        accum, rem = divmod(target, per_step)
        if accum < 1:
            sys.exit(f"ERROR: batch {args.batch} x {args.gpus} GPUs = {per_step} "
                     f"already exceeds the global batch {target}. Lower --batch.")
        if rem:
            sys.exit(f"ERROR: {per_step} does not divide {target} evenly "
                     f"(accum {accum}, remainder {rem}). Global batch would be "
                     f"{per_step * accum}, changing the optimisation problem. "
                     f"Pick a batch size that divides {target}.")
        c["optimization"]["accum_steps"] = accum

    c["data"]["batch_size"] = args.batch
    if args.measure:
        c["optimization"].update(epochs=1, ipe=50, warmup=0, anneal=0)
        if args.stage == 1:
            # accum 1 so ms/sample is measured on a single micro-batch, not
            # averaged across an accumulation loop.
            c["optimization"]["accum_steps"] = 1
            accum = 1

    yaml.safe_dump(c, open(path, "w"), sort_keys=False)

    print(f"{rel}")
    print(f"  batch_size   {args.batch}")
    print(f"  accum_steps  {accum}")
    print(f"  global batch {args.batch * args.gpus * accum}"
          f"{'  (paper)' if args.batch * args.gpus * accum == target else ''}")
    if args.measure:
        print(f"  MEASURE MODE epochs 1, ipe 50, warmup 0, anneal 0 -- timing only")

    # The hang guard prepare_configs.py applies, repeated here: __len__ is the
    # number of paired EPISODES and the loader uses drop_last=True, so too few
    # pairs gives len(loader)==0 and a silent infinite hang.
    root = c["data"]["dataset"]
    if os.path.isdir(root):
        pairs = sum(len([f for f in os.listdir(os.path.join(root, t, "franka"))
                         if f.endswith((".hdf5", ".h5"))])
                    for t in os.listdir(root)
                    if os.path.isdir(os.path.join(root, t, "franka")))
        per_rank = pairs // args.gpus
        batches = per_rank // args.batch
        print(f"  episode pairs {pairs} -> {per_rank}/rank -> {batches} batches/epoch")
        if batches == 0:
            sys.exit(f"\n*** FATAL: batch {args.batch} > {per_rank} pairs per rank.\n"
                     f"    len(loader)==0 with drop_last=True -- training HANGS "
                     f"silently.\n    Collect more episodes or lower --batch.")
        if batches < 20:
            print(f"\n  WARNING: only {batches} batches/epoch. The loader will "
                  f"rebuild often\n  ('Exhausted data loaders. Refreshing...'), "
                  f"inflating the `data` timing.\n  Want >20 for a trustworthy "
                  f"throughput measurement.")


if __name__ == "__main__":
    main()

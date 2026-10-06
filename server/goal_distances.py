#!/usr/bin/env python3
"""Calibrate deploy.l1_threshold from the latent distances deploy actually measures.

deploy advances to the next reference goal only when the current view is within
l1_threshold (L1 between layer-normed latents) of the goal. If the threshold sits
below the distance between two views a few mm apart, the robot can never advance
-- the first oracle rollout sat at 0.322 against 0.30 while 6.6 mm from its goal.

Encodes with deploy's own path (build_world_model + WorldModel.encode, same
weights, transform and dtype), so these are the numbers deploy compares.

  python server/goal_distances.py                                  # first val scene
  python server/goal_distances.py --rollout rollouts/oracle_norot/ep0

Prints, along one demo:
  d(t, t+1)   one raw frame apart (50 ms) -- the floor: near-identical views
  d(t, t+4)   one goal step apart -- what each advance has to close
  d(0, k)     distance from the start as the demo progresses
and, with --rollout, how close the restored scene's first saved view is to the
demo's frames (does the scene restore render like the recording?).
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch
import yaml
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from app.vjepa_2_1_dreamer_ac.deploy import build_world_model, latent_l1_distance  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", default="configs/inference/deploy_vjepa_2_1.yaml",
                   help="deploy config (written by prepare_deploy_config.py)")
    p.add_argument("--demo", default=None,
                   help="franka demo h5; default: first recovered scene's val demo")
    p.add_argument("--rollout", default=None, help="rollouts/<name>/epN to compare")
    p.add_argument("--stride", type=int, default=4, help="raw frames per goal step")
    args = p.parse_args()

    params = yaml.safe_load(open(os.path.join(REPO, args.fname)))
    params["deploy"]["goal_mode"] = "oracle"          # encoder + predictor only
    wm, dtype, mixed = build_world_model(params)

    demo = args.demo
    if demo is None:
        scenes = sorted(glob.glob(os.path.join(REPO, "data", "scenes", "*", "*")))
        if not scenes:
            sys.exit("no recovered scenes; pass --demo")
        task, name = scenes[0].split(os.sep)[-2:]
        demo = os.path.join(REPO, "data", "val", task, "franka", name + ".hdf5")
    with h5py.File(demo, "r") as f:
        imgs = np.asarray(f["observations/images/right_shoulder_rgb"])
    T = len(imgs)
    print(f"demo {demo}: {T} frames\n")

    def enc(img):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype, enabled=mixed):
            return wm.encode(np.asarray(img, dtype=np.uint8))

    z = [enc(imgs[t]) for t in range(T)]
    d = lambda a, b: latent_l1_distance(a, b)

    same = d(z[0], enc(imgs[0]))
    one = [d(z[t], z[t + 1]) for t in range(T - 1)]
    step = [d(z[t], z[t + args.stride]) for t in range(T - args.stride)]
    print(f"re-encode same frame        {same:.4f}   (sanity: ~0)")
    print(f"d(t, t+1)   1 raw frame     median {np.median(one):.4f}  min {min(one):.4f}  max {max(one):.4f}")
    print(f"d(t, t+{args.stride})   1 goal step     median {np.median(step):.4f}  min {min(step):.4f}  max {max(step):.4f}")
    print("d(0, k):    " + "  ".join(f"{k}:{d(z[0], z[k]):.3f}" for k in range(0, T, args.stride)))

    if args.rollout:
        pngs = sorted(glob.glob(os.path.join(REPO, args.rollout, "**", "*.png"), recursive=True))
        if not pngs:
            print(f"\nno frames under {args.rollout}")
        else:
            r0 = enc(Image.open(pngs[0]).convert("RGB"))
            near = [(d(r0, z[k]), k) for k in range(min(T, 3 * args.stride))]
            best = min(near)
            print(f"\nrollout first saved view ({os.path.basename(pngs[0])}) vs demo frames 0..{len(near) - 1}:")
            print("  " + "  ".join(f"{k}:{v:.3f}" for v, k in sorted(near, key=lambda x: x[1])))
            print(f"  closest: demo frame {best[1]} at {best[0]:.4f}")

    lo, hi = np.median(one), np.median(step)
    print(f"\nl1_threshold must sit ABOVE the near-identical floor (~{lo:.3f}) or the robot "
          f"can never advance,\nand below a full goal step (~{hi:.3f}) or it advances without "
          f"moving. Midpoint: {(lo + hi) / 2:.3f}")


if __name__ == "__main__":
    main()

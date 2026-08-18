#!/usr/bin/env python3
"""Write the deploy/rollout config.

Keeps the upstream MPC settings, which ARE the paper's:

    samples: 200   cem_steps: 50   topk: 10   rollout: 1

That is 200 x 50 = 10,000 predictor forwards per environment step. It is slow
by design -- do not trim it to make rollouts finish faster unless you intend to
report a reduced-compute result, in which case say so explicitly.

Paths are repo-relative and the reference demo is auto-selected from the
held-out split, so there is normally nothing to edit.

Usage:
  python server/prepare_deploy_config.py
  python server/prepare_deploy_config.py --task push_button --max-steps 60
"""
import argparse
import glob
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = "configs/inference/deploy_vjepa_2_1.yaml"
DEFAULT_DEPLOY_CKPT = os.path.join(REPO, "exp", "stage2_deploy.pt")
DEFAULT_STAGE1_CKPT = os.path.join(REPO, "exp", "stage1", "latest.pt")
HELD_OUT = os.path.join(REPO, "data", "val")
OUT_FOLDER = os.path.join(REPO, "exp", "deploy")
REFERENCE_ROBOT = "sawyer"   # source embodiment providing the one-shot demo


def pick_reference(task, camera):
    """First held-out episode of the reference embodiment, sorted for determinism."""
    if task:
        tasks = [task]
    else:
        tasks = sorted(d for d in os.listdir(HELD_OUT)
                       if os.path.isdir(os.path.join(HELD_OUT, d))) if os.path.isdir(HELD_OUT) else []
        if not tasks:
            sys.exit(f"ERROR: no task dirs under {HELD_OUT}. Run server/split_dataset.py first.")

    pattern = os.path.join(HELD_OUT, tasks[0], REFERENCE_ROBOT, "*.hdf5")
    hits = sorted(glob.glob(pattern))
    if not hits:
        sys.exit(f"ERROR: no reference episodes matching {pattern}\n"
                 f"Run server/split_dataset.py first.")
    return tasks[0], hits[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=None, help="defaults to the first held-out task")
    p.add_argument("--camera", default="right_shoulder_rgb")
    p.add_argument("--max-steps", type=int, default=40, help="env steps per episode")
    p.add_argument("--deploy-ckpt", default=DEFAULT_DEPLOY_CKPT)
    p.add_argument("--stage1-ckpt", default=DEFAULT_STAGE1_CKPT)
    args = p.parse_args()

    task, reference_h5 = pick_reference(args.task, args.camera)

    path = os.path.join(REPO, CFG)
    c = yaml.safe_load(open(path))

    # Guard: these are the paper's MPC values. If upstream differs, stop and look.
    mpc = c["deploy"]["mpc"]
    expected = {"samples": 200, "cem_steps": 50, "topk": 10, "rollout": 1}
    for k, v in expected.items():
        if mpc.get(k) != v:
            sys.exit(f"ERROR: {CFG}: deploy.mpc.{k} is {mpc.get(k)!r}, expected {v!r}. "
                     f"Upstream changed -- confirm before running.")

    c["folder"] = OUT_FOLDER
    c["meta"]["pretrain_checkpoint"] = args.deploy_ckpt
    c["meta"]["dreamer_predictor_checkpoint"] = args.stage1_ckpt
    c["deploy"]["reference_h5"] = reference_h5
    c["deploy"]["image_key"] = f"observations/images/{args.camera}"
    c["deploy"]["max_steps"] = args.max_steps
    # mpc block left exactly as upstream.

    yaml.safe_dump(c, open(path, "w"), sort_keys=False)

    print(f"wrote {CFG}")
    print(f"  task           {task}")
    for lbl, val in (("deploy ckpt", args.deploy_ckpt),
                     ("stage 1 ckpt", args.stage1_ckpt),
                     ("reference h5", reference_h5)):
        print(f"  {lbl:14s} {val}  [{'ok' if os.path.exists(val) else 'MISSING'}]")
    print(f"  mpc            {dict(mpc)}  (paper values, unchanged)")
    print(f"  max_steps      {args.max_steps}")
    print(f"\nReference demo is held-out ({REFERENCE_ROBOT}); the policy drives the")
    print(f"other embodiment. That cross-embodiment gap is the thing being measured.")


if __name__ == "__main__":
    main()

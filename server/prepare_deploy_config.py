#!/usr/bin/env python3
"""Write the deploy/rollout config.

Keeps the upstream repo's MPC search settings:

    samples: 200   cem_steps: 50   topk: 10   rollout: 1

maxnorm is OURS (template value, or --maxnorm), sized to the data's action
scale; upstream shipped 0.1.

*** THESE ARE NOT THE PAPER'S VALUES. *** Checked against arXiv 2605.20811 on
2026-09-02: Algorithm 1 (Appendix B) defines population size N, elites K,
momentum beta, horizon H and iterations L, but states NO numeric values for any
of them. Tables C.1-C.3 cover architecture, not planning. `maxnorm` does not
appear in the paper at all. An earlier version of this file claimed otherwise;
it was wrong.

So these are upstream defaults, and changing them is a deviation from the REPO,
not from the paper. That matters: at our data scale the per-step action deltas
are 0.004-0.011 m, so maxnorm 0.1 puts the entire CEM search 10-20x outside the
training distribution -- measured 14/14 IK failures at 0.1 versus 0/4 at 0.01.
Tuning an unspecified hyperparameter to the data is legitimate; say what you
used and why.

samples x cem_steps = 10,000 predictor forwards per environment step, and MPC
replans every step. Note rollout: 1 means a ONE-STEP planning horizon -- greedy,
not lookahead.

Paths are repo-relative and the reference demo is auto-selected from the
held-out split, so there is normally nothing to edit.

Usage:
  python server/prepare_deploy_config.py
  python server/prepare_deploy_config.py --task push_button --max-steps 60
"""
import argparse
import glob
import os
import subprocess
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
    # Measured values go in as flags, not yaml edits: the template is read from
    # git HEAD, so an uncommitted edit to it would silently change nothing.
    p.add_argument("--maxnorm", type=float, default=None,
                   help="override mpc.maxnorm (measure it: RUNBOOK_LAB step 6)")
    p.add_argument("--l1-threshold", type=float, default=None,
                   help="override deploy.l1_threshold (RUNBOOK_LAB step 11)")
    args = p.parse_args()

    task, reference_h5 = pick_reference(args.task, args.camera)

    path = os.path.join(REPO, CFG)
    # Read the UPSTREAM config from git HEAD, not the working tree. This script
    # overwrites the same file it reads, so reading the working tree makes it
    # non-idempotent: a second run sees the first run's output and the MPC guard
    # below fires against our own values -- or worse, against a deliberate
    # local MPC reduction, reporting it as "upstream changed".
    blob = subprocess.run(["git", "show", f"HEAD:{CFG}"],
                          cwd=REPO, capture_output=True, text=True)
    if blob.returncode != 0:
        sys.exit(f"ERROR: cannot read {CFG} from git HEAD:\n{blob.stderr.strip()}\n"
                 f"Are you inside the repo, and is {CFG} committed?")
    c = yaml.safe_load(blob.stdout)

    # Guard against upstream drift -- NOT a claim that these are the paper's
    # values (the paper states none; see the module docstring).
    mpc = c["deploy"]["mpc"]
    expected = {"samples": 200, "cem_steps": 50, "topk": 10, "rollout": 1}
    for k, v in expected.items():
        if mpc.get(k) != v:
            sys.exit(f"ERROR: {CFG}: deploy.mpc.{k} is {mpc.get(k)!r}, expected the "
                     f"upstream default {v!r}. Upstream changed, or the committed "
                     f"config was edited -- confirm before running.")

    c["folder"] = OUT_FOLDER
    c["meta"]["pretrain_checkpoint"] = args.deploy_ckpt
    c["meta"]["dreamer_predictor_checkpoint"] = args.stage1_ckpt
    c["deploy"]["reference_h5"] = reference_h5
    c["deploy"]["image_key"] = f"observations/images/{args.camera}"
    c["deploy"]["max_steps"] = args.max_steps
    # mpc guard keys left exactly as upstream; maxnorm/l1_threshold come from
    # the committed template (see its comments).

    # Temporal alignment, derived rather than typed: the reference stride must
    # equal ONE action's span, which training fixed as ceil(data_fps / fps)
    # raw frames (train.py passes tubelet_size=1). Read the capture rate from
    # the reference demo itself and the planning rate from the stage 2 config
    # training actually used, so the two cannot drift apart.
    import h5py
    from math import ceil
    with h5py.File(reference_h5, "r") as _f:
        _dt = _f.attrs.get("dt")
    if _dt is None:
        sys.exit(f"ERROR: {reference_h5} has no 'dt' attr; cannot derive the "
                 f"reference frame rate.")
    src_fps = int(round(1.0 / float(_dt)))
    s2 = yaml.safe_load(open(os.path.join(REPO, "configs/train/vjepa_2_1_dreamer_ac.yaml")))
    plan_fps = int(s2["data"]["fps"])
    train_data_fps = int(s2["data"]["data_fps"])
    if train_data_fps != src_fps:
        sys.exit(f"ERROR: stage 2 was configured with data_fps={train_data_fps} but "
                 f"the reference demo is {src_fps} Hz. Re-run prepare_configs.py "
                 f"(it derives data_fps from the data) and RETRAIN stage 2 -- a "
                 f"checkpoint trained at a different stride cannot be deployed here.")
    if ceil(src_fps / plan_fps) != src_fps // plan_fps:
        sys.exit(f"ERROR: {src_fps} Hz is not a multiple of {plan_fps} fps; training "
                 f"(ceil) and deploy (floor) would disagree on the action stride.")
    c["deploy"]["ref_data_fps"] = src_fps
    c["deploy"]["ref_target_fps"] = plan_fps
    if args.maxnorm is not None:
        c["deploy"]["mpc"]["maxnorm"] = args.maxnorm
    if args.l1_threshold is not None:
        c["deploy"]["l1_threshold"] = args.l1_threshold

    yaml.safe_dump(c, open(path, "w"), sort_keys=False)

    print(f"wrote {CFG}")
    print(f"  task           {task}")
    for lbl, val in (("deploy ckpt", args.deploy_ckpt),
                     ("stage 1 ckpt", args.stage1_ckpt),
                     ("reference h5", reference_h5)):
        print(f"  {lbl:14s} {val}  [{'ok' if os.path.exists(val) else 'MISSING'}]")
    print(f"  mpc            {dict(mpc)}")
    print(f"                 ^ samples/cem_steps/topk/rollout: upstream defaults, NOT in")
    print(f"                   the paper. maxnorm: ours, sized to the data (not upstream's 0.1)")
    print(f"  max_steps      {args.max_steps}")
    print(f"  maxnorm        {c['deploy']['mpc']['maxnorm']}   "
          f"l1_threshold {c['deploy']['l1_threshold']}")
    print(f"  frame_skip     {src_fps} Hz // {plan_fps} fps = {src_fps // plan_fps} raw "
          f"frames = one training action")
    print(f"\nReference demo is held-out ({REFERENCE_ROBOT}); the policy drives the")
    print(f"other embodiment. That cross-embodiment gap is the thing being measured.")


if __name__ == "__main__":
    main()

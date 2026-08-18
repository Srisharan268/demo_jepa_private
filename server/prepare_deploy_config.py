#!/usr/bin/env python3
"""Write the deploy/rollout config.

Keeps the upstream MPC settings, which ARE the paper's:

    samples: 200   cem_steps: 50   topk: 10   rollout: 1

That is 200 x 50 = 10,000 predictor forwards per environment step. It is slow
by design -- do not trim it to make rollouts finish faster unless you intend to
report a reduced-compute result, in which case say so explicitly.

Changes only paths, the reference demo, and max_steps.

Usage:  python server/prepare_deploy_config.py
"""
import os
import sys

import yaml

# ----------------------------------------------------------------------------
# EDIT THESE
# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEPLOY_CKPT = "/path/to/stage2_deploy.pt"        # from make_deploy_ckpt.py
STAGE1_CKPT = "/path/to/exp/stage1/latest.pt"    # Dreamer Predictor
REFERENCE_H5 = "/path/to/held_out/<task>/sawyer/episode0.hdf5"   # one-shot prompt
OUT_FOLDER = "/path/to/exp/deploy"

CAMERA = "right_shoulder_rgb"
MAX_STEPS = 40          # env steps per episode before giving up
# ----------------------------------------------------------------------------

CFG = "configs/inference/deploy_vjepa_2_1.yaml"
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
c["meta"]["pretrain_checkpoint"] = DEPLOY_CKPT
c["meta"]["dreamer_predictor_checkpoint"] = STAGE1_CKPT
c["deploy"]["reference_h5"] = REFERENCE_H5
c["deploy"]["image_key"] = f"observations/images/{CAMERA}"
c["deploy"]["max_steps"] = MAX_STEPS
# mpc block left exactly as upstream.

yaml.safe_dump(c, open(path, "w"), sort_keys=False)

print(f"wrote {CFG}")
for label, val in (("deploy ckpt", DEPLOY_CKPT),
                   ("stage 1 ckpt", STAGE1_CKPT),
                   ("reference h5", REFERENCE_H5)):
    print(f"  {label:14s} {val}  [{'ok' if os.path.exists(val) else 'MISSING'}]")
print(f"  mpc            {mpc}  (paper values, unchanged)")
print(f"  max_steps      {MAX_STEPS}")

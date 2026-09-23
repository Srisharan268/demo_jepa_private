#!/usr/bin/env python3
"""Does stage 1 produce a usable cross-embodiment goal? Nothing has tested this.

Stage 1 is the whole premise: dreamer(x_t^franka, y_t^sawyer, y_{t+n}^sawyer)
-> where the franka should be at t+n. Stage 2 is then trained to match that
output (paper SS3.3: L_plan = ||F_wm(z,s,a) - z_goal||^2), and CEM plans toward
it. Everything routes through this one module, and the only evidence it works
is that its training loss fell and it did not collapse.

Two questions, both unanswered until now:

1. IS THE GOAL CORRECT? Compare dreamer output against the true franka latent
   at t+n, with baselines that make the number interpretable:
     identity  - L1(x_t, x_{t+n}): what you get by predicting "no change".
                 The goal must beat this or it is worse than doing nothing.
     chance    - L1(x_t, random other frame): the scale of "unrelated".

2. DOES IT USE THE REFERENCE? Recompute the goal with the sawyer reference
   swapped for a DIFFERENT episode's, and measure how far the output moves.
   This is the load-bearing one. If the goal barely changes, then
   z_goal ~= f(x_t) -- a function of the franka context alone, which is also
   stage 2's input. The action would then be redundant BY CONSTRUCTION, no
   matter how the data is sampled, and the observed action-invariance of stage 2
   would be a downstream symptom of a stage 1 failure rather than a data-scale
   problem.

Also sweeps the temporal gap n, because stage 1 trains on
target_idx = randint(current+1, episode_len) -- a uniform random gap averaging
~30 frames on a 93-frame demo -- while deploy asks it for a small gap. If goal
quality degrades sharply at small n, deploy is querying it out of distribution.

Uses TRAINING image geometry by default (scale 1.777). deploy.py builds its
transform at scale 1.0, which the encoder never saw in training; that mismatch
is itself under investigation, so --deploy-transform reproduces it.

No simulator, forward passes only:

  cd ~/Demo-JEPA && WANDB_MODE=disabled python server/dreamer_test.py
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from app.vjepa_2_1_dreamer_ac.deploy import build_world_model  # noqa: E402
from app.vjepa_2_1_dreamer_ac.transforms import make_transforms  # noqa: E402


def l1(a, b):
    return float(F.l1_loss(a.flatten(1), b.flatten(1)).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", default="configs/inference/deploy_vjepa_2_1.yaml")
    p.add_argument("--train-cfg", default="configs/train/vjepa_2_1_dreamer_ac.yaml")
    p.add_argument("--task", default="push_button")
    p.add_argument("--gaps", type=int, nargs="+", default=[4, 10, 20, 40])
    p.add_argument("--samples", type=int, default=4, help="start frames per gap")
    p.add_argument("--deploy-transform", action="store_true",
                   help="use deploy's scale 1.0 instead of training's 1.777")
    args = p.parse_args()

    params = yaml.safe_load(open(os.path.join(REPO, args.fname)))
    tcfg = yaml.safe_load(open(os.path.join(REPO, args.train_cfg)))
    aug = tcfg.get("data_aug", {})
    crop = int(tcfg["data"]["crop_size"])
    img_key = params["deploy"].get("image_key",
                                   "observations/images/right_shoulder_rgb")

    root = os.path.join(REPO, "data", "val", args.task)
    fr = sorted(glob.glob(os.path.join(root, "franka", "*.hdf5")))
    if len(fr) < 2:
        sys.exit(f"ERROR: need >=2 held-out episodes under {root}/franka")

    world_model, dtype, mixed = build_world_model(params)

    if not args.deploy_transform:
        sc = tuple(aug.get("random_resize_scale", [1.0, 1.0]))
        ar = tuple(aug.get("random_resize_aspect_ratio", [1.0, 1.0]))
        world_model.transform = make_transforms(
            random_horizontal_flip=False, random_resize_aspect_ratio=ar,
            random_resize_scale=sc, reprob=0.0, auto_augment=False,
            motion_shift=False, crop_size=crop)
        print(f"\ntransform: TRAINING geometry  scale={sc} aspect={ar}")
    else:
        print("\ntransform: DEPLOY geometry  scale=(1.0, 1.0)")

    def load(path):
        with h5py.File(path, "r") as f:
            return np.asarray(f[img_key])

    ep_a = os.path.basename(fr[0])
    fa = load(fr[0])
    sa = load(os.path.join(root, "sawyer", ep_a))
    # A different episode's reference: same task, different scene. Swapping to
    # this is the test of whether the reference is used at all.
    ep_b = os.path.basename(fr[1])
    sb = load(os.path.join(root, "sawyer", ep_b))

    print(f"episode {ep_a}: {len(fa)} frames   (alt reference: {ep_b})\n")

    enc = world_model.encode
    rows = []
    for gap in args.gaps:
        acc = {k: [] for k in ("goal", "ident", "chance", "moved", "refswap")}
        for s in range(args.samples):
            t = 5 + s * max(1, (len(fa) - gap - 10) // max(args.samples, 1))
            if t + gap >= min(len(fa), len(sa), len(sb)):
                continue
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype, enabled=mixed):
                x_t, x_next = enc(fa[t]), enc(fa[t + gap])
                y_t, y_next = enc(sa[t]), enc(sa[t + gap])
                goal = world_model.forward_dreamer_predictor(x_t, y_t, y_next)
                # same franka context, reference from a DIFFERENT episode
                gb = world_model.forward_dreamer_predictor(
                    x_t, enc(sb[t]), enc(sb[t + gap]))
                rnd = enc(fa[(t + gap + len(fa) // 2) % len(fa)])

                acc["goal"].append(l1(goal, x_next))
                acc["ident"].append(l1(x_t, x_next))
                acc["chance"].append(l1(x_t, rnd))
                acc["moved"].append(l1(goal, x_t))
                acc["refswap"].append(l1(goal, gb))
        if not acc["goal"]:
            continue
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        rows.append((gap, m))
        print(f"gap {gap:3d}:  goal->true {m['goal']:.4f}   identity {m['ident']:.4f}"
              f"   chance {m['chance']:.4f}")
        print(f"          goal moved from x_t by {m['moved']:.4f}"
              f"   reference-swap moves goal by {m['refswap']:.4f}")

    if not rows:
        sys.exit("ERROR: no samples ran")

    print(f"\n{'=' * 68}")
    print(f"{'gap':>5} {'goal':>8} {'identity':>9} {'beats id':>9} "
          f"{'ref-swap':>9} {'ref/move':>9}")
    for gap, m in rows:
        beats = "YES" if m["goal"] < m["ident"] else "no"
        frac = m["refswap"] / (m["moved"] + 1e-9)
        print(f"{gap:>5} {m['goal']:>8.4f} {m['ident']:>9.4f} {beats:>9} "
              f"{m['refswap']:>9.4f} {frac:>8.2f}x")
    print(f"{'=' * 68}")

    best = min(rows, key=lambda r: r[1]["goal"])[1]
    any_beats = any(m["goal"] < m["ident"] for _g, m in rows)
    ref_frac = float(np.mean([m["refswap"] / (m["moved"] + 1e-9) for _g, m in rows]))

    print("GOAL QUALITY:", end=" ")
    if any_beats:
        print("the goal beats the identity baseline at some gap.")
        print("  => stage 1 predicts forward motion, not just a copy of x_t.")
    else:
        print("the goal NEVER beats identity.")
        print("  => stage 1's output is worse than predicting 'no change'. The goal")
        print("     CEM chases is not a useful target, and stage 2 was trained to")
        print("     match it. Stage 1 is the problem, not data scale.")

    print("REFERENCE USE:", end=" ")
    if ref_frac < 0.1:
        print(f"swapping the reference moves the goal only {ref_frac:.2f}x as far")
        print("  as the goal moved from x_t at all.")
        print("  => the dreamer LARGELY IGNORES the demonstration. z_goal ~= f(x_t),")
        print("     a function of stage 2's own input, so the action is redundant BY")
        print("     CONSTRUCTION. This would explain stage 2's action-invariance")
        print("     WITHOUT any appeal to fps or data scale -- and the fps fix would")
        print("     not help. Stage 1 has to be fixed first.")
    else:
        print(f"swapping the reference moves the goal {ref_frac:.2f}x as far as the")
        print("  goal moved from x_t.")
        print("  => the dreamer DOES condition on the demonstration. Stage 1 is")
        print("     carrying cross-embodiment signal; look elsewhere for stage 2's")
        print("     action-invariance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

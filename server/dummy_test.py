#!/usr/bin/env python3
"""Is the world model sound, or is the GOAL wrong? Run CEM without the dreamer.

Rollouts saturate: xyz action components pin to +/-maxnorm on nearly every step,
across maxnorm in {0.1, 0.01, 0.005, 0.02}, with the scene correctly restored,
and the commanded heading is INVERTED in y and z against the demo (demo descends
70 cm to the button; the policy climbs and wedges in 15 mm). Two causes remain
and they need opposite fixes:

  A. the GOAL is wrong. Stage 1 trains on pairs (o_k, o_{k+n}) with
     n = randint(current+1, episode_len) -- a uniform random gap over the rest
     of the episode, averaging ~30 frames on a 93-frame demo. Deploy asks it for
     a 2-frame gap, far outside that distribution, so its output is
     unconstrained. Fix is on the stage 1 side; no deploy tuning helps.

  B. the WORLD MODEL is wrong -- stage 2's xyz action conditioning is degenerate,
     so the predicted latent barely depends on the xyz action, the cost surface
     is near-flat, and CEM walks to a corner. Fix is stage 2; stage 1 is
     irrelevant.

This removes the dreamer from the loop: CEM runs against the TRUE next franka
frame from a held-out episode. That goal is correctly scaled and reachable by
construction -- it is literally where the arm went next.

  sensible action  -> world model and CEM are fine, cause is A
  still saturates  -> cause is B

STRIDE. One training action spans 2 raw frames (dataset.py: primary_states
[::frameskip], frameskip = tubelet_size = 2, fstp = ceil(data_fps/fps) = 1 for
the checkpoint we have). That is a derivation from source, so --sweep re-checks
it empirically instead of trusting it: if the model behaves best at some other
stride, the derivation is wrong. Note this is NOT the dreamer's frame_skip --
that one is bypassed here entirely, which is the whole point.

Beyond the CEM result this asks the model to SCORE three actions against the
same goal, which separates a planner failure from a model failure:

  l1(GT) < l1(CEM)  -> the model knows the GT action is better and CEM failed to
                       find it: a PLANNER problem (note cem_utils samples with
                       std == the clip bound, putting ~32% of the mass on the
                       boundary)
  l1(CEM) < l1(GT)  -> the model genuinely prefers an action the training data
                       never contains: a MODEL error

No simulator, seconds per timestep:

  cd ~/Demo-JEPA && WANDB_MODE=disabled python server/dummy_test.py
  cd ~/Demo-JEPA && WANDB_MODE=disabled python server/dummy_test.py --sweep 1 2 4 6
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
from scipy.spatial.transform import Rotation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from app.vjepa_2_1_dreamer_ac.cem_utils import cem, compute_new_pose  # noqa: E402
from app.vjepa_2_1_dreamer_ac.deploy import build_world_model  # noqa: E402


def quaternion_to_euler(poses):
    """dataset.py's version, so the GT action matches training exactly."""
    if poses.shape[-1] == 7:
        return poses
    xyz, quat, grip = poses[:, :3], poses[:, 3:7], poses[:, -1:]
    euler = np.stack([Rotation.from_quat(q).as_euler("xyz") for q in quat])
    return np.concatenate([xyz, euler, grip], axis=1)


def poses_to_diffs(poses):
    """dataset.py's version. Rotation deltas are R[t+1] @ R[t].T, not a subtraction."""
    xyz, thetas = poses[:, :3], poses[:, 3:6]
    mats = [Rotation.from_euler("xyz", t).as_matrix() for t in thetas]
    xyz_diff = xyz[1:] - xyz[:-1]
    ang = np.stack([Rotation.from_matrix(mats[t + 1] @ mats[t].T).as_euler("xyz")
                    for t in range(len(mats) - 1)])
    return np.concatenate([xyz_diff, ang, poses[:, -1:][1:]], axis=1)


def make_wm(world_model):
    """The same one-step closure WorldModel.__call__ hands to cem()."""
    def wm(reps, actions, poses):
        B, T, N_T, D = reps.size()
        flat = reps.flatten(1, 2)
        nxt = world_model.predictor(flat, actions, poses)[:, -world_model.tokens_per_frame:]
        if world_model.normalize_reps:
            nxt = F.layer_norm(nxt, (nxt.size(-1),))
        nxt = nxt.view(B, 1, N_T, D)
        nxt_pose = compute_new_pose(poses[:, -1:], actions[:, -1:],
                                    abs_gripper=world_model.abs_gripper)
        return nxt, nxt_pose
    return wm


def score(wm, rep, pose, action, goal_rep):
    """The model's own cost for one action: l1 between its prediction and the goal."""
    a = torch.as_tensor(np.asarray(action, dtype=np.float32),
                        dtype=rep.dtype, device=rep.device).view(1, 1, -1)
    nxt, _ = wm(rep.repeat(1, 1, 1, 1), a, pose.repeat(1, 1, 1))
    return float(F.l1_loss(nxt.flatten(1), goal_rep.repeat(1, 1, 1, 1).flatten(1)).item())


def run_stride(world_model, wm, mpc, dtype, mixed, imgs, qpos, stride, start, steps, maxnorm):
    euler = quaternion_to_euler(qpos)
    gt = poses_to_diffs(euler[::stride])       # action k spans k*stride -> (k+1)*stride

    rows = []
    for n in range(steps):
        k = start + n
        i = k * stride
        if i + stride >= len(imgs) or k >= len(gt):
            break

        with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed):
            rep = world_model.encode(imgs[i])
            goal = world_model.encode(imgs[i + stride])
            pose = quaternion_to_euler(qpos[i][None, :].astype(np.float64))
            pose = torch.from_numpy(pose).to(dtype).to(world_model.device)

            action, _ = cem(context_frame=rep, context_pose=pose, goal_frame=goal,
                            world_model=wm, **mpc)
            action = action[0].detach().float().cpu().numpy().reshape(-1)

            l1_cem = score(wm, rep, pose, action, goal)
            l1_gt = score(wm, rep, pose, gt[k], goal)
            l1_zero = score(wm, rep, pose, np.zeros_like(gt[k]), goal)

        g, a = gt[k][:3], action[:3]
        ng, na = np.linalg.norm(g), np.linalg.norm(a)
        cos = float(np.dot(g, a) / (ng * na + 1e-12))
        sat = int((np.abs(a) >= 0.99 * maxnorm).sum())
        rows.append(dict(i=i, gt=g, act=a, ng=ng, na=na, cos=cos, sat=sat,
                         l1_cem=l1_cem, l1_gt=l1_gt, l1_zero=l1_zero))

        print(f"  [frame {i:3d}] GT {g.round(5)} |{ng:.5f}|   "
              f"CEM {a.round(5)} |{na:.5f}|  sat {sat}/3  cos {cos:+.3f}")
        print(f"             cost  CEM {l1_cem:.5f}   GT {l1_gt:.5f}   zero {l1_zero:.5f}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", default="configs/inference/deploy_vjepa_2_1.yaml")
    p.add_argument("--episode", default=None, help="default: first held-out franka episode")
    p.add_argument("--steps", type=int, default=5, help="timesteps per stride")
    p.add_argument("--start", type=int, default=3, help="first action index")
    p.add_argument("--train-transform", action="store_true",
                   help="use training geometry (scale 1.777); deploy builds 1.0, "
                        "which the encoder never saw in training")
    p.add_argument("--sweep", type=int, nargs="+", default=None,
                   help="strides to try (default: just 2, the derived training stride)")
    args = p.parse_args()

    ep = args.episode or (sorted(glob.glob(
        os.path.join(REPO, "data", "val", "*", "franka", "*.hdf5"))) or [None])[0]
    if ep is None:
        sys.exit("ERROR: no held-out franka episode under data/val")

    params = yaml.safe_load(open(os.path.join(REPO, args.fname)))
    img_key = params["deploy"].get("image_key", "observations/images/right_shoulder_rgb")
    maxnorm = float(params["deploy"]["mpc"]["maxnorm"])

    with h5py.File(ep, "r") as f:
        imgs = np.asarray(f[img_key])
        qpos = np.asarray(f["observations/qpos"], dtype=np.float64)

    print(f"episode {os.path.basename(ep)}: {len(imgs)} frames")
    print(f"maxnorm {maxnorm}  (saturated = |a| >= {0.99 * maxnorm:.5f})")
    print("goal = the TRUE next franka frame; the dreamer is NOT in this loop.\n")

    world_model, dtype, mixed = build_world_model(params)
    if args.train_transform:
        import yaml as _y
        from app.vjepa_2_1_dreamer_ac.transforms import make_transforms
        _a = _y.safe_load(open(os.path.join(
            REPO, "configs/train/vjepa_2_1_dreamer_ac.yaml"))).get("data_aug", {})
        _sc = tuple(_a.get("random_resize_scale", [1.0, 1.0]))
        _ar = tuple(_a.get("random_resize_aspect_ratio", [1.0, 1.0]))
        world_model.transform = make_transforms(
            random_horizontal_flip=False, random_resize_aspect_ratio=_ar,
            random_resize_scale=_sc, reprob=0.0, auto_augment=False,
            motion_shift=False, crop_size=int(params["data"]["crop_size"]))
        print(f"transform: TRAINING geometry scale={_sc} aspect={_ar}\n")
    wm = make_wm(world_model)
    mpc = dict(world_model.mpc_args)
    mpc["abs_gripper"] = world_model.abs_gripper

    summary = []
    for stride in (args.sweep or [2]):
        print(f"\n--- stride {stride} raw frames "
              f"{'(derived training stride)' if stride == 2 else ''} ---")
        rows = run_stride(world_model, wm, mpc, dtype, mixed, imgs, qpos,
                          stride, args.start, args.steps, maxnorm)
        if not rows:
            print("  (no timesteps ran)")
            continue
        summary.append((stride,
                        np.mean([r["sat"] for r in rows]) / 3.0,
                        np.mean([r["cos"] for r in rows]),
                        np.mean([r["na"] / (r["ng"] + 1e-12) for r in rows]),
                        np.mean([r["l1_cem"] < r["l1_gt"] for r in rows])))

    if not summary:
        sys.exit("ERROR: nothing ran -- check --start against the episode length")

    print(f"\n{'=' * 70}")
    print(f"{'stride':>7} {'saturated':>10} {'cos vs GT':>10} {'|a|/|gt|':>9} "
          f"{'model prefers CEM':>19}")
    for s, sat, cos, ratio, pref in summary:
        print(f"{s:>7} {sat * 100:>9.0f}% {cos:>+10.3f} {ratio:>8.1f}x {pref * 100:>18.0f}%")
    print(f"{'=' * 70}")

    # Verdict from the derived stride if it ran, else the best available.
    s, sat, cos, ratio, pref = next((r for r in summary if r[0] == 2), summary[0])
    print(f"verdict from stride {s}:")
    if sat < 0.35 and cos > 0.3:
        print("  NOT saturated and heading agrees with ground truth against a true,")
        print("  reachable goal.")
        print("  => world model and CEM are SOUND. The rollout goal is the problem")
        print("     (cause A): the dreamer's output horizon. Fix stage 1.")
    elif pref > 0.5:
        print("  Saturated, and the model SCORES its own saturated action better than")
        print("  the ground-truth action it was trained on.")
        print("  => the WORLD MODEL is wrong (cause B), not the planner. Stage 2's")
        print("     action conditioning is the fix; stage 1 will not help.")
    else:
        print("  Saturated, but the model correctly scores the GT action better.")
        print("  => a PLANNER problem: the model is usable and CEM is not finding the")
        print("     good action. cem_utils samples with std == the clip bound, so ~32%")
        print("     of the mass starts on the boundary. Try std < maxnorm first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Random-action "play" episodes for the stage 0 world model (target robot only).

Why this exists. The AC predictor plans from ONE frame of context. In a planner
demo, the next motion from a frame is predictable from the scene alone (the arm
heads for the button), so a model trained only on demos can minimise its loss
while ignoring the action -- exactly how the previous stage 2 failed (swapping
in shuffled actions changed its loss by 1e-5). Here the motion is random, so the
next frame can only be predicted by reading the action. And "many different
actions from one state" is precisely what CEM asks the model at deploy.

Each episode: reset the task scene (fresh random layout), then a sequence of
random end-effector actions. One action spans --stride recorded frames, the
same span one training action covers (ceil(20 Hz / 5 fps) = 4). Actions mix a
drift toward a random waypoint (to cover the workspace, including near the
task objects) with independent noise (so an action cannot be guessed from the
motion before it). Bounds match deploy's CEM sampling: xyz and rotation within
+/- maxnorm per action (0.05 m / 0.05 rad), absolute gripper, occasionally
toggled.

Frames are recorded exactly as the paired collector records them -- one
move_to_pose per frame, settled -- and qpos is the ACHIEVED pose, so the
dataloader's stride-4 pose differences are true labels whatever the IK did.

Output: <save_path>/<task>/franka/play_v<v>_m<master>_<i>.hdf5. Keep it OUT of
data/train (the paired loaders would look for a sawyer twin). prepare_configs.py
links data/play/<task>/franka into stage 0's dataset.

  python play.py --save_path ../../data/play --task push_button --episodes 50 \
      --seed_master 1 --headless
"""
import argparse
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK

from action_utils import demo_to_ee_actions, set_seed
from camera import apply_fixed_camera
from config import DEFAULT_CAMERA_JSON
from env_utils import make_env, get_task_env
from io_utils import ensure_dir, save_demo_h5
from observation import build_obs_config
from retarget import move_to_pose, pair_seed, robot_dir


def workspace_bounds(start_pos, args):
    """Box the tip may roam: RLBench's 'workspace' table shape if present, else
    a box around the start pose. z never goes below table + margin."""
    try:
        from pyrep.objects.shape import Shape
        ws = Shape("workspace")
        minx, maxx, miny, maxy, _, maxz = ws.get_bounding_box()
        p = ws.get_position()
        m = 0.05
        lo = np.array([p[0] + minx + m, p[1] + miny + m, p[2] + maxz + args.z_margin])
        hi = np.array([p[0] + maxx - m, p[1] + maxy - m, start_pos[2] + 0.05])
        src = "workspace shape"
    except Exception as err:          # noqa: BLE001 -- any failure -> fallback box
        lo = start_pos - args.box
        hi = start_pos + np.array([args.box, args.box, 0.05])
        src = f"start +/- {args.box} m ({type(err).__name__})"
    return lo, np.maximum(hi, lo + 0.02), src


def sample_action(rng, cur_pos, cur_rot, start_rot, waypoint, args):
    """One random 7-d delta action [dxyz, drpy, gripper-toggle]."""
    to_wp = waypoint - cur_pos
    dist = np.linalg.norm(to_wp)
    speed = rng.uniform(0.0, args.xyz_max)
    drift = to_wp / max(dist, 1e-6) * min(dist, speed)
    noise = rng.uniform(-args.xyz_max, args.xyz_max, 3) * args.noise_frac
    dxyz = np.clip(drift + noise, -args.xyz_max, args.xyz_max)

    if rng.random() < args.rot_prob:
        drpy = rng.uniform(-args.rot_max, args.rot_max, 3)
    else:
        drpy = np.zeros(3)
    # Keep orientation within rot_limit of the start: far rotations mostly
    # produce IK failures, not data. Steer back when outside.
    rel = (cur_rot * start_rot.inv()).as_rotvec()
    if np.linalg.norm(rel) > args.rot_limit:
        drpy = np.clip(-rel, -args.rot_max, args.rot_max)

    toggle = rng.random() < args.grip_prob
    return dxyz, drpy, toggle


def run_episode(task_env, env, rng, args):
    obs = task_env.get_observation()
    start_pos = np.asarray(obs.gripper_pose[:3], dtype=np.float64)
    start_rot = Rotation.from_quat(obs.gripper_pose[3:7])
    lo, hi, src = workspace_bounds(start_pos, args)

    frames, commanded = [], []
    waypoint = rng.uniform(lo, hi)
    wp_age, n_fail, consec_fail = 0, 0, 0
    grip = float(obs.gripper_open > 0.5)

    for _ in range(args.actions):
        cur_pos = np.asarray(obs.gripper_pose[:3], dtype=np.float64)
        cur_rot = Rotation.from_quat(obs.gripper_pose[3:7])
        if np.linalg.norm(waypoint - cur_pos) < 0.02 or wp_age >= args.wp_every:
            waypoint, wp_age = rng.uniform(lo, hi), 0
        wp_age += 1

        dxyz, drpy, toggle = sample_action(rng, cur_pos, cur_rot, start_rot, waypoint, args)
        # Reflect out-of-bounds components instead of pinning to the wall.
        end = cur_pos + dxyz
        out = (end < lo) | (end > hi)
        dxyz[out] = -dxyz[out]
        dxyz = np.clip(cur_pos + dxyz, lo, hi) - cur_pos
        if toggle:
            grip = 1.0 - grip

        # Same convention as server.py: target = cur + delta, R = R(delta) * R(cur),
        # split evenly over the frames one training action spans.
        ok = True
        for f in range(1, args.stride + 1):
            frac = f / args.stride
            q = (Rotation.from_euler("xyz", drpy * frac) * cur_rot).as_quat()
            target = np.concatenate([cur_pos + dxyz * frac, q, [grip]]).astype(np.float32)
            try:
                obs, _, _, _ = move_to_pose(task_env, env, target, args.settle_pos_eps,
                                            args.settle_ori_eps_deg, args.settle_max_steps)
            except Exception:         # IK failure: InvalidActionError and friends
                ok = False
                break
            frames.append(obs)
            commanded.append(target)

        if ok:
            consec_fail = 0
        else:
            n_fail += 1
            consec_fail += 1
            waypoint, wp_age = rng.uniform(lo, hi), 0
            obs = task_env.get_observation()
            if consec_fail >= args.max_consec_fail:
                break

    return frames, np.asarray(commanded, dtype=np.float32), n_fail, src


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--save_path", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--robot", default="panda", help="target embodiment (paper: Franka)")
    p.add_argument("--variation", type=int, default=0)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed_master", type=int, required=True,
                   help="distinct per parallel process; also distinct from the paired run's")
    p.add_argument("--actions", type=int, default=40, help="actions per episode")
    p.add_argument("--stride", type=int, default=4,
                   help="recorded frames per action = ceil(data_fps / fps) of training")
    p.add_argument("--xyz_max", type=float, default=0.05, help="= deploy mpc.maxnorm")
    p.add_argument("--rot_max", type=float, default=0.05, help="rad, = deploy maxnorm (use_rpy)")
    p.add_argument("--rot_prob", type=float, default=0.5)
    p.add_argument("--rot_limit", type=float, default=0.5, help="rad from start orientation")
    p.add_argument("--noise_frac", type=float, default=0.6)
    p.add_argument("--wp_every", type=int, default=8)
    p.add_argument("--grip_prob", type=float, default=0.05)
    p.add_argument("--box", type=float, default=0.25)
    p.add_argument("--z_margin", type=float, default=0.03)
    p.add_argument("--max_consec_fail", type=int, default=10)
    p.add_argument("--min_frames", type=int, default=33, help="stage 0/2 need 8*4+1")
    p.add_argument("--image_size", nargs=2, type=int, default=[640, 480])
    p.add_argument("--renderer", choices=["opengl", "opengl3"], default="opengl3")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--arm_max_velocity", type=float, default=1.0)
    p.add_argument("--arm_max_acceleration", type=float, default=4.0)
    p.add_argument("--settle_pos_eps", type=float, default=1e-3)
    p.add_argument("--settle_ori_eps_deg", type=float, default=2.0)
    p.add_argument("--settle_max_steps", type=int, default=40)
    p.add_argument("--camera_json", default=DEFAULT_CAMERA_JSON)
    args = p.parse_args()

    out_dir = os.path.join(args.save_path, args.task, robot_dir(args.robot))
    ensure_dir(out_dir)

    env = make_env(
        robot_setup=args.robot,
        obs_config=build_obs_config(width=args.image_size[0], height=args.image_size[1],
                                    renderer=args.renderer),
        arm_mode=EndEffectorPoseViaIK(),
        headless=args.headless,
        arm_max_velocity=args.arm_max_velocity,
        arm_max_acceleration=args.arm_max_acceleration,
        static_positions=False,
        dt=args.dt,
    )
    n_ok = n_short = 0
    try:
        task_env = get_task_env(env, args.task)
        for i in range(args.episodes):
            name = f"play_v{args.variation}_m{args.seed_master}_{i:04d}"
            final = os.path.join(out_dir, name + ".hdf5")
            if os.path.exists(final):
                print(f"[SKIP] {name}")
                continue
            t0 = time.time()
            # Offset the variation slot so play seeds never equal paired seeds
            # even if the same master is reused.
            seed = pair_seed(args.seed_master, 10_000 + args.variation, i)
            set_seed(seed)
            rng = np.random.default_rng(seed)
            task_env.set_variation(args.variation)
            task_env.reset()
            apply_fixed_camera(args.camera_json)

            frames, commanded, n_fail, src = run_episode(task_env, env, rng, args)
            if len(frames) < args.min_frames:
                n_short += 1
                print(f"[SHORT] {name}: {len(frames)} frames ({n_fail} IK failures) -- not saved")
                continue

            achieved = demo_to_ee_actions(frames)
            tmp = final + ".tmp"
            save_demo_h5(
                out_h5=tmp,
                demo_obs=frames,
                actions=achieved,
                commanded=commanded,
                image_hw=(args.image_size[1], args.image_size[0]),
                camera_names=["right_shoulder_rgb"],
                sim=True,
                attrs={
                    "kind": "play",
                    "task": args.task,
                    "robot": args.robot,
                    "variation": int(args.variation),
                    "episode_seed_used": int(seed),
                    "seed_master": int(args.seed_master),
                    "dt": float(args.dt),
                    "renderer": args.renderer,
                    "stride": int(args.stride),
                    "xyz_max": float(args.xyz_max),
                    "rot_max": float(args.rot_max),
                    "ik_failures": int(n_fail),
                    "bounds": src,
                },
            )
            os.replace(tmp, final)
            n_ok += 1
            step = np.linalg.norm(achieved[args.stride:, :3] - achieved[:-args.stride, :3], axis=1)
            print(f"[OK] {name}: {len(frames)} frames, {n_fail} IK fail, per-action move "
                  f"median {np.median(step) * 1000:.0f} mm / max {step.max() * 1000:.0f} mm, "
                  f"bounds: {src}, {time.time() - t0:.0f}s")
    finally:
        env.shutdown()
    print(f"[DONE] saved {n_ok}, short {n_short}, in {out_dir}")


if __name__ == "__main__":
    main()

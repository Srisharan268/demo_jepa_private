"""Paired cross-embodiment data collection.

One live demo is generated on the SOURCE robot; its end-effector trajectory is
then replayed through IK on every robot in `cfg.robots`, in the same scene
(same variation, same RNG state). Each robot's replay is one episode.

Output is written directly in the layout the training dataloader reads -- no
separate flattening step, which is what previously threw away the scene files:

    <save_path>/<task>/franka/<name>.hdf5
    <save_path>/<task>/sawyer/<name>.hdf5

Each HDF5 is self-contained: full collection config, the RNG state that
reproduces its scene, and per-robot replay quality are stored in its attributes.

Fixes relative to the previous collector (all measured on the 402-pair dataset):

  * pair_seed was LINEAR in (master, pair), so chunks run with consecutive
    masters produced identical seeds: chunk i's pair 1 == chunk i+1's pair 0.
    150 of 402 episodes were bit-identical copies, and 30 of 40 val episodes
    duplicated train episodes. Seeds are now hashed.
  * qpos was built from the COMMANDED source poses, identical for every robot
    (40/40 pairs had franka qpos == sawyer qpos). Each robot's episode now
    records the pose that robot actually ACHIEVED, so action labels describe
    what its images show. Commanded poses are kept as `commanded_pose`.
  * Task success was never checked; a replay that never completed the task was
    saved. Every robot's replay must now succeed.
  * A killed run left a partial pair directory that later runs skipped as
    "existing". Files are written as .tmp and renamed only when the whole pair
    has succeeded.
  * total_episodes < variations silently collected nothing.
"""
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaIK

from action_utils import demo_to_ee_actions, quat_distance_deg, set_seed
from camera import apply_fixed_camera
from config import RetargetConfig
from env_utils import make_env, get_task_env, get_variation_count
from io_utils import ensure_dir, save_demo_h5
from observation import build_obs_config

# RLBench calls the arm "panda"; the training dataloader reads a "franka/" dir.
ROBOT_DIR = {"panda": "franka"}


def robot_dir(robot: str) -> str:
    return ROBOT_DIR.get(robot, robot)


def task_succeeded(task_env) -> bool:
    """RLBench evaluates success only inside task_env.step(). The settle steps in
    move_to_pose bypass that, so a press completed while settling would be
    missed -- ask the task directly as well."""
    task = getattr(task_env, "_task", None)
    return bool(task is not None and task.success()[0])


def move_to_pose(
    task_env,
    env,
    target: np.ndarray,
    pos_eps: float,
    ori_eps_deg: float,
    max_steps: int,
) -> Tuple[object, bool, float, float]:
    """
    Move the end-effector to a target pose [x, y, z, qx, qy, qz, qw, gripper_open].

    Returns (obs, success, pos_err_m, ori_err_deg). The errors are measured on
    the FINAL observation: settling can time out without reaching the target,
    and previously that was invisible.
    """
    target_pos = target[:3]
    target_quat = target[3:7]

    obs, reward, _ = task_env.step(target)
    success = reward > 0

    for _ in range(max_steps):
        env._pyrep.step()
        obs = task_env.get_observation()

        pos_ok = np.linalg.norm(obs.gripper_pose[:3] - target_pos) <= pos_eps
        ori_ok = quat_distance_deg(target_quat, obs.gripper_pose[3:]) <= ori_eps_deg

        if pos_ok and ori_ok:
            break

    pos_err = float(np.linalg.norm(obs.gripper_pose[:3] - target_pos))
    ori_err = float(quat_distance_deg(target_quat, obs.gripper_pose[3:]))
    return obs, bool(success or task_succeeded(task_env)), pos_err, ori_err


def generate_source_episode(
    cfg: RetargetConfig,
    variation_index: int,
    seed: int,
) -> Tuple[np.ndarray, object, int, List[str]]:
    """
    Generate one live demo from the source robot.

    Returns (actions, rng_state, actual_var_idx, descriptions). RLBench's
    get_demos only returns demos that succeeded, so the source trajectory is a
    successful one. The RNG path (set_seed -> reset -> get_demos) is unchanged
    from the previous collector, so server/recover_rng.py can still regenerate
    old datasets' scenes from their stored seeds.
    """
    obs_cfg = build_obs_config(
        width=cfg.image_width,
        height=cfg.image_height,
        renderer=cfg.renderer,
    )

    env = make_env(
        robot_setup=cfg.source_robot,
        obs_config=obs_cfg,
        arm_mode=JointVelocity(),
        headless=cfg.headless,
        arm_max_velocity=cfg.arm_max_velocity,
        arm_max_acceleration=cfg.arm_max_acceleration,
        static_positions=cfg.static_positions,
        dt=cfg.dt,
    )

    try:
        task_env = get_task_env(env, cfg.task)
        task_env.set_variation(variation_index)

        set_seed(seed)

        last_error = None

        for _ in range(cfg.max_demo_attempts):
            try:
                descriptions, _ = task_env.reset()
                apply_fixed_camera(cfg.camera_json)

                [demo] = task_env.get_demos(
                    amount=1,
                    live_demos=True,
                    max_attempts=cfg.max_demo_attempts,
                )

                actions = demo_to_ee_actions(demo)
                rng_state = demo.random_seed
                actual_var_idx = demo._observations[0].misc["variation_index"]

                return actions, rng_state, int(actual_var_idx), list(descriptions or [])

            except Exception as err:
                last_error = err

        raise RuntimeError(f"failed to get source demo: {last_error}")

    finally:
        env.shutdown()


def replay_episode(
    cfg: RetargetConfig,
    robot_setup: str,
    variation_index: int,
    rng_state,
    actions: np.ndarray,
) -> Tuple[List, Dict]:
    """
    Replay source EE-pose actions on one robot under the same scene seed.

    Returns (demo_obs, info) where info records whether the task was completed
    and how closely this robot tracked the commanded poses.
    """
    obs_cfg = build_obs_config(
        width=cfg.image_width,
        height=cfg.image_height,
        renderer=cfg.renderer,
    )

    env = make_env(
        robot_setup=robot_setup,
        obs_config=obs_cfg,
        arm_mode=EndEffectorPoseViaIK(),
        headless=cfg.headless,
        arm_max_velocity=cfg.arm_max_velocity,
        arm_max_acceleration=cfg.arm_max_acceleration,
        static_positions=cfg.static_positions,
        dt=cfg.dt,
    )

    try:
        task_env = get_task_env(env, cfg.task)
        task_env.set_variation(variation_index)

        np.random.set_state(rng_state)

        task_env.reset()
        apply_fixed_camera(cfg.camera_json)

        demo_obs, pos_errs, ori_errs = [], [], []
        success = False

        for action in actions:
            obs, ok, pos_err, ori_err = move_to_pose(
                task_env=task_env,
                env=env,
                target=action,
                pos_eps=cfg.settle_pos_eps,
                ori_eps_deg=cfg.settle_ori_eps_deg,
                max_steps=cfg.settle_max_steps,
            )
            demo_obs.append(obs)
            pos_errs.append(pos_err)
            ori_errs.append(ori_err)
            success = success or ok

        info = {
            "success": bool(success),
            "max_pos_err": float(np.max(pos_errs)),
            "mean_pos_err": float(np.mean(pos_errs)),
            "max_ori_err_deg": float(np.max(ori_errs)),
        }
        return demo_obs, info

    finally:
        env.shutdown()


def pair_seed(
    master_seed: int,
    variation_index: int,
    pair_index: int,
) -> int:
    """
    Deterministic, collision-free seed for each (master, variation, pair).

    The previous version fed master + variation * 1000003 + pair into
    RandomState, which is LINEAR: masters one apart collide one pair apart.
    SeedSequence hashes the tuple, so any change to any component gives an
    unrelated seed.
    """
    ss = np.random.SeedSequence([int(master_seed), int(variation_index), int(pair_index)])
    return int(ss.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)


def episode_name(cfg: RetargetConfig, variation_index: int, pair_index: int) -> str:
    # The master is in the name so parallel chunks (job-array tasks) never
    # write to the same file.
    return f"variation{variation_index}_m{cfg.seed_master}_{pair_index:04d}"


def episode_path(cfg: RetargetConfig, robot: str, name: str) -> str:
    return os.path.join(cfg.save_path, cfg.task, robot_dir(robot), name + ".hdf5")


def build_meta(
    cfg: RetargetConfig,
    actual_var_idx: int,
    seed: int,
    descriptions: List[str],
) -> dict:
    """Everything needed to reproduce this episode, stored IN each HDF5."""
    return {
        "task": cfg.task,
        "variation": int(actual_var_idx),
        "seed_master": int(cfg.seed_master),
        "episode_seed_used": int(seed),
        "robots": list(cfg.robots),
        "source_robot": cfg.source_robot,
        "renderer": cfg.renderer,
        "image_size": [int(cfg.image_width), int(cfg.image_height)],
        "dt": float(cfg.dt),
        "static_positions": bool(cfg.static_positions),
        "arm_max_velocity": float(cfg.arm_max_velocity),
        "arm_max_acceleration": float(cfg.arm_max_acceleration),
        "max_demo_attempts": int(cfg.max_demo_attempts),
        "retries_per_pair": int(cfg.retries_per_pair),
        "max_track_err": float(cfg.max_track_err),
        "camera_json": cfg.camera_json,
        "settle": {
            "pos_eps": float(cfg.settle_pos_eps),
            "ori_eps_deg": float(cfg.settle_ori_eps_deg),
            "max_steps": int(cfg.settle_max_steps),
        },
        "descriptions": list(descriptions),
        "qpos": "pose ACHIEVED by this robot; commanded source pose is commanded_pose",
    }


def rng_state_attrs(rng_state) -> Dict:
    """numpy's legacy RNG state as HDF5-storable attributes.

    server.py restores a scene from this state; storing it in the episode file
    means it cannot be lost the way the separate rng_state.pkl files were.
    server/recover_rng.py writes it back out as rng_state.pkl for --episode_dir.
    """
    algo, keys, pos, has_gauss, cached = rng_state
    return {
        "rng_state_algo": str(algo),
        "rng_state_keys": np.asarray(keys, dtype=np.uint32),
        "rng_state_pos": int(pos),
        "rng_state_has_gauss": int(has_gauss),
        "rng_state_cached_gaussian": float(cached),
    }


def save_pair(
    cfg: RetargetConfig,
    name: str,
    source_actions: np.ndarray,
    rng_state,
    actual_var_idx: int,
    seed: int,
    demos: List[List],
    infos: List[Dict],
    descriptions: List[str],
) -> None:
    """Write every robot's episode as .tmp, then rename them all.

    A pair only becomes visible once every robot's file is complete, so an
    interrupted run never leaves something a later run mistakes for finished.
    """
    meta = build_meta(cfg, actual_var_idx, seed, descriptions)
    rng = rng_state_attrs(rng_state)
    written = []

    try:
        for robot, demo_obs, info in zip(cfg.robots, demos, infos):
            final = episode_path(cfg, robot, name)
            tmp = final + ".tmp"
            written.append((tmp, final))

            save_demo_h5(
                out_h5=tmp,
                demo_obs=demo_obs,
                actions=demo_to_ee_actions(demo_obs),   # ACHIEVED poses
                commanded=source_actions,
                image_hw=(cfg.image_height, cfg.image_width),
                camera_names=["right_shoulder_rgb"],
                sim=True,
                attrs={
                    "task": cfg.task,
                    "robot": robot,
                    "source_robot": cfg.source_robot,
                    "variation": int(actual_var_idx),
                    "episode_seed_used": int(seed),
                    "seed_master": int(cfg.seed_master),
                    "renderer": cfg.renderer,
                    "dt": float(cfg.dt),
                    "static_positions": bool(cfg.static_positions),
                    "replay_success": bool(info["success"]),
                    "replay_max_pos_err": info["max_pos_err"],
                    "replay_mean_pos_err": info["mean_pos_err"],
                    "replay_max_ori_err_deg": info["max_ori_err_deg"],
                    "meta": meta,
                    **rng,
                },
            )

        for tmp, final in written:
            os.replace(tmp, final)

    except BaseException:
        for tmp, _ in written:
            if os.path.exists(tmp):
                os.remove(tmp)
        raise


def run_collection(cfg: RetargetConfig) -> None:
    if not cfg.task:
        raise ValueError("Please specify --task")

    for robot in cfg.robots:
        ensure_dir(os.path.join(cfg.save_path, cfg.task, robot_dir(robot)))

    # Leftovers from a killed run: never complete, never to be trusted.
    for stale in glob.glob(os.path.join(cfg.save_path, cfg.task, "*", "*.hdf5.tmp")):
        os.remove(stale)

    variation_count = get_variation_count(
        task_name=cfg.task,
        robot_setup=cfg.source_robot,
        headless=cfg.headless,
        static_positions=cfg.static_positions,
        dt=cfg.dt,
    )

    num_variations = (
        variation_count
        if cfg.variations < 0
        else min(cfg.variations, variation_count)
    )

    if num_variations <= 0:
        raise ValueError(f"Invalid num_variations={num_variations}")

    pairs_per_variation = cfg.total_episodes // num_variations
    if pairs_per_variation <= 0:
        raise ValueError(
            f"--total_episodes {cfg.total_episodes} < {num_variations} variations: "
            f"that is 0 pairs per variation and would collect nothing. Raise "
            f"--total_episodes or limit --variations.")

    print(f"[INFO] task={cfg.task}")
    print(f"[INFO] variation_count={variation_count}")
    print(f"[INFO] num_variations={num_variations}")
    print(f"[INFO] pairs_per_variation={pairs_per_variation}")
    print(f"[INFO] seed_master={cfg.seed_master}")
    print(f"[INFO] robots={list(cfg.robots)} -> dirs {[robot_dir(r) for r in cfg.robots]}")
    print(f"[INFO] save_path={cfg.save_path}")

    n_ok = n_skip = n_give_up = 0

    for variation_index in range(num_variations):
        for pair_index in range(pairs_per_variation):
            name = episode_name(cfg, variation_index, pair_index)

            if all(os.path.exists(episode_path(cfg, r, name)) for r in cfg.robots):
                print(f"[SKIP] complete: {name}")
                n_skip += 1
                continue

            seed = pair_seed(
                master_seed=cfg.seed_master,
                variation_index=variation_index,
                pair_index=pair_index,
            )

            success = False

            for attempt in range(1, cfg.retries_per_pair + 1):
                try:
                    actions, rng_state, actual_var_idx, descriptions = generate_source_episode(
                        cfg=cfg,
                        variation_index=variation_index,
                        seed=seed,
                    )

                    demos, infos = [], []
                    for robot in cfg.robots:
                        demo_obs, info = replay_episode(
                            cfg=cfg,
                            robot_setup=robot,
                            variation_index=int(actual_var_idx),
                            rng_state=rng_state,
                            actions=actions,
                        )
                        if not info["success"]:
                            raise RuntimeError(f"{robot} replay did not complete the task")
                        if cfg.max_track_err > 0 and info["max_pos_err"] > cfg.max_track_err:
                            raise RuntimeError(
                                f"{robot} replay deviated {info['max_pos_err'] * 1000:.1f} mm "
                                f"from the demo (limit {cfg.max_track_err * 1000:.1f} mm)")
                        demos.append(demo_obs)
                        infos.append(info)

                    save_pair(
                        cfg=cfg,
                        name=name,
                        source_actions=actions,
                        rng_state=rng_state,
                        actual_var_idx=int(actual_var_idx),
                        seed=seed,
                        demos=demos,
                        infos=infos,
                        descriptions=descriptions,
                    )

                    errs = ", ".join(f"{r} {i['max_pos_err'] * 1000:.1f}mm"
                                     for r, i in zip(cfg.robots, infos))
                    print(f"[OK] {cfg.task}/{name} saved, attempt={attempt}, "
                          f"max tracking err: {errs}")

                    success = True
                    n_ok += 1
                    break

                except Exception as err:
                    print(
                        f"[FAIL] {cfg.task}/{name} "
                        f"attempt {attempt}/{cfg.retries_per_pair}: {err}"
                    )

            if not success:
                n_give_up += 1
                print(
                    f"[GIVE UP] {cfg.task}/{name} "
                    f"after {cfg.retries_per_pair} attempts"
                )

    attempted = n_ok + n_give_up
    print(f"[DONE] saved {n_ok}, skipped (already complete) {n_skip}, gave up {n_give_up}"
          + (f"  -> yield {100 * n_ok / attempted:.0f}%" if attempted else ""))

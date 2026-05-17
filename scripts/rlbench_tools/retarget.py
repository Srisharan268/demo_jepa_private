import os
import shutil
from typing import List, Tuple

import numpy as np

from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaIK
from rlbench.backend.const import (
    VARIATIONS_FOLDER,
    EPISODES_FOLDER,
    EPISODE_FOLDER,
)
from rlbench.observation_config import ObservationConfig

from action_utils import demo_to_ee_actions, quat_distance_deg, set_seed
from camera import apply_fixed_camera
from config import RetargetConfig
from env_utils import make_env, get_task_env, get_variation_count
from io_utils import (
    ensure_dir,
    save_demo_h5,
    save_meta,
    save_rng_state,
    save_variation_descriptions,
)
from observation import build_obs_config


def move_to_pose(
    task_env,
    env,
    target: np.ndarray,
    pos_eps: float,
    ori_eps_deg: float,
    max_steps: int,
):
    """
    Move the current robot end-effector to a target EE pose.

    target:
        [x, y, z, qx, qy, qz, qw, gripper_open]
    """
    target_pos = target[:3]
    target_quat = target[3:7]

    obs, _, _ = task_env.step(target)

    for _ in range(max_steps):
        env._pyrep.step()
        obs = task_env.get_observation()

        pos_ok = np.linalg.norm(obs.gripper_pose[:3] - target_pos) <= pos_eps
        ori_ok = quat_distance_deg(target_quat, obs.gripper_pose[3:]) <= ori_eps_deg

        if pos_ok and ori_ok:
            break

    return obs


def generate_source_episode(
    cfg: RetargetConfig,
    variation_index: int,
    seed: int,
) -> Tuple[np.ndarray, object, int]:
    """
    Generate one live demo from the source robot.

    Return:
        actions:
            EE-pose actions extracted from source demo, shape (T, 8)

        rng_state:
            RLBench random state for reproducing the same scene

        actual_var_idx:
            actual variation index
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
                task_env.reset()
                apply_fixed_camera(cfg.camera_json)

                [demo] = task_env.get_demos(
                    amount=1,
                    live_demos=True,
                    max_attempts=cfg.max_demo_attempts,
                )

                actions = demo_to_ee_actions(demo)
                rng_state = demo.random_seed
                actual_var_idx = demo._observations[0].misc["variation_index"]

                return actions, rng_state, int(actual_var_idx)

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
) -> List:
    """
    Replay source EE-pose actions on one robot under the same scene seed.
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

        demo_obs = []

        for action in actions:
            obs = move_to_pose(
                task_env=task_env,
                env=env,
                target=action,
                pos_eps=cfg.settle_pos_eps,
                ori_eps_deg=cfg.settle_ori_eps_deg,
                max_steps=cfg.settle_max_steps,
            )
            demo_obs.append(obs)

        return demo_obs

    finally:
        env.shutdown()


def load_variation_descriptions(
    cfg: RetargetConfig,
    variation_index: int,
):
    """
    Reset the source environment once to obtain variation descriptions.
    These descriptions are kept as environment/initial-condition related metadata.
    """
    env = make_env(
        robot_setup=cfg.source_robot,
        obs_config=ObservationConfig(),
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

        descriptions, _ = task_env.reset()
        apply_fixed_camera(cfg.camera_json)

        return descriptions

    finally:
        env.shutdown()


def pair_seed(
    master_seed: int,
    variation_index: int,
    pair_index: int,
) -> int:
    """
    Deterministic seed for each variation/pair.
    This makes resuming easier and more reproducible.
    """
    rng = np.random.RandomState(
        master_seed + variation_index * 1000003 + pair_index
    )
    return int(rng.randint(0, 2**31 - 1))


def build_pair_root(
    cfg: RetargetConfig,
    variation_index: int,
    pair_index: int,
) -> str:
    return os.path.join(
        cfg.save_path,
        cfg.task,
        VARIATIONS_FOLDER % variation_index,
        f"{pair_index:04d}",
    )


def build_episode_roots(
    cfg: RetargetConfig,
    pair_root: str,
) -> List[str]:
    return [
        os.path.join(
            pair_root,
            EPISODES_FOLDER,
            EPISODE_FOLDER % 0,
            robot,
        )
        for robot in cfg.robots
    ]


def build_meta(
    cfg: RetargetConfig,
    actual_var_idx: int,
    seed: int,
) -> dict:
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
        "camera_json": cfg.camera_json,
        "settle": {
            "pos_eps": float(cfg.settle_pos_eps),
            "ori_eps_deg": float(cfg.settle_ori_eps_deg),
            "max_steps": int(cfg.settle_max_steps),
        },
        "data_format": {
            "type": "hdf5",
            "file_name": "episode.hdf5",
            "qpos": "observations/qpos",
            "qvel": "observations/qvel",
            "action": "action",
            "image": "observations/images/right_shoulder_rgb",
        },
    }


def save_pair_data(
    cfg: RetargetConfig,
    pair_root: str,
    actions: np.ndarray,
    rng_state,
    actual_var_idx: int,
    seed: int,
    demos: List[List],
) -> None:
    """
    Save one paired episode.

    Files saved:
        pair_root/meta.json
        pair_root/rng_state.pkl
        pair_root/variation_descriptions.pkl
        pair_root/episodes/episode0/<robot>/episode.hdf5
    """
    episode_roots = build_episode_roots(cfg, pair_root)

    for robot, episode_root, demo_obs in zip(cfg.robots, episode_roots, demos):
        ensure_dir(episode_root)

        out_h5 = os.path.join(episode_root, "episode.hdf5")

        save_demo_h5(
            out_h5=out_h5,
            demo_obs=demo_obs,
            actions=actions,
            image_hw=(cfg.image_height, cfg.image_width),
            camera_names=["right_shoulder_rgb"],
            sim=True,
            attrs={
                "task": cfg.task,
                "robot": robot,
                "source_robot": cfg.source_robot,
                "variation": int(actual_var_idx),
                "episode_seed_used": int(seed),
                "renderer": cfg.renderer,
                "dt": float(cfg.dt),
                "static_positions": bool(cfg.static_positions),
            },
        )

    meta = build_meta(
        cfg=cfg,
        actual_var_idx=actual_var_idx,
        seed=seed,
    )

    save_meta(pair_root, meta)
    save_rng_state(pair_root, rng_state)

    try:
        descriptions = load_variation_descriptions(cfg, int(actual_var_idx))
        save_variation_descriptions(pair_root, descriptions)
    except Exception as err:
        print(f"[WARN] failed to save variation descriptions: {err}")


def run_collection(cfg: RetargetConfig) -> None:
    if not cfg.task:
        raise ValueError("Please specify --task")

    ensure_dir(cfg.save_path)

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

    print(f"[INFO] task={cfg.task}")
    print(f"[INFO] variation_count={variation_count}")
    print(f"[INFO] num_variations={num_variations}")
    print(f"[INFO] pairs_per_variation={pairs_per_variation}")
    print(f"[INFO] save_path={cfg.save_path}")

    for variation_index in range(num_variations):
        for pair_index in range(pairs_per_variation):
            pair_root = build_pair_root(
                cfg=cfg,
                variation_index=variation_index,
                pair_index=pair_index,
            )

            if os.path.exists(pair_root):
                print(f"[SKIP] existing: {pair_root}")
                continue

            seed = pair_seed(
                master_seed=cfg.seed_master,
                variation_index=variation_index,
                pair_index=pair_index,
            )

            success = False

            for attempt in range(1, cfg.retries_per_pair + 1):
                os.makedirs(pair_root, exist_ok=True)

                try:
                    actions, rng_state, actual_var_idx = generate_source_episode(
                        cfg=cfg,
                        variation_index=variation_index,
                        seed=seed,
                    )

                    demos = []

                    for robot in cfg.robots:
                        demo_obs = replay_episode(
                            cfg=cfg,
                            robot_setup=robot,
                            variation_index=int(actual_var_idx),
                            rng_state=rng_state,
                            actions=actions,
                        )
                        demos.append(demo_obs)

                    save_pair_data(
                        cfg=cfg,
                        pair_root=pair_root,
                        actions=actions,
                        rng_state=rng_state,
                        actual_var_idx=int(actual_var_idx),
                        seed=seed,
                        demos=demos,
                    )

                    print(
                        f"[OK] {cfg.task}/v{variation_index}/pair{pair_index} "
                        f"saved, attempt={attempt}"
                    )

                    success = True
                    break

                except Exception as err:
                    shutil.rmtree(pair_root, ignore_errors=True)
                    print(
                        f"[FAIL] {cfg.task}/v{variation_index}/pair{pair_index} "
                        f"attempt {attempt}/{cfg.retries_per_pair}: {err}"
                    )

            if not success:
                print(
                    f"[GIVE UP] {cfg.task}/v{variation_index}/pair{pair_index} "
                    f"after {cfg.retries_per_pair} attempts"
                )
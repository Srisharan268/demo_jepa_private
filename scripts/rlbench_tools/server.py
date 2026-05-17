#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import socket

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaIK,
    EndEffectorPoseViaPlanning,
)
from rlbench.backend.exceptions import InvalidActionError

from action_utils import quat_distance_deg
from camera import apply_fixed_camera
from config import DEFAULT_CAMERA_JSON
from env_utils import make_env, get_task_env
from observation import build_obs_config
from io_utils import ensure_dir
from socket_utils import send_msg, recv_msg


# ============================================================
# Pair metadata utilities
# ============================================================

def find_pair_root(path: str) -> str:
    """
    Robustly infer pair_root from:
        pair_root
        pair_root/episodes/episode0
        pair_root/episodes/episode0/panda
        pair_root/episodes/episode0/panda/episode.hdf5

    pair_root is detected by meta.json or rng_state.pkl.
    """
    cur = os.path.abspath(path)

    if os.path.isfile(cur):
        cur = os.path.dirname(cur)

    for _ in range(12):
        meta_path = os.path.join(cur, "meta.json")
        rng_path = os.path.join(cur, "rng_state.pkl")

        if os.path.exists(meta_path) or os.path.exists(rng_path):
            return cur

        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    raise FileNotFoundError(
        f"Could not infer pair_root from path: {path}. "
        f"Expected to find meta.json or rng_state.pkl in its parent chain."
    )


def load_meta(pair_root: str) -> dict:
    meta_path = os.path.join(pair_root, "meta.json")

    if not os.path.exists(meta_path):
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rng_state(pair_root: str):
    rng_path = os.path.join(pair_root, "rng_state.pkl")

    if not os.path.exists(rng_path):
        raise FileNotFoundError(f"rng_state.pkl not found: {rng_path}")

    with open(rng_path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Observation packing and optional image saving
# ============================================================

def pack_observation(
    obs,
    reward: float,
    terminated: bool,
    send_images: bool = True,
) -> dict:
    data = {
        "gripper_pose": obs.gripper_pose.tolist()
        if hasattr(obs, "gripper_pose") and obs.gripper_pose is not None
        else None,

        "gripper_open": float(obs.gripper_open)
        if hasattr(obs, "gripper_open") and obs.gripper_open is not None
        else None,

        "joint_positions": obs.joint_positions.tolist()
        if hasattr(obs, "joint_positions") and obs.joint_positions is not None
        else None,

        "task_low_dim_state": obs.task_low_dim_state.tolist()
        if hasattr(obs, "task_low_dim_state") and obs.task_low_dim_state is not None
        else None,

        "reward": float(reward),
        "terminated": bool(terminated),
        "right_shoulder_rgb": None,
    }

    if data["gripper_open"] is not None:
        data["gripper_open"] = 1.0 if data["gripper_open"] > 0.5 else 0.0

    if (
        send_images
        and hasattr(obs, "right_shoulder_rgb")
        and obs.right_shoulder_rgb is not None
    ):
        data["right_shoulder_rgb"] = obs.right_shoulder_rgb

    return data


def save_rgb_image(rgb_np, save_path: str) -> None:
    if rgb_np is None:
        return

    img = np.asarray(rgb_np)

    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    Image.fromarray(img).save(save_path)


def create_episode_dir(save_root: str, episode_counter: int) -> str:
    ep_dir = os.path.join(save_root, f"episode_{episode_counter:04d}")
    ensure_dir(ep_dir)
    return ep_dir


def save_step_image(
    obs,
    save_dir: str,
    step_idx: int,
    suffix: str = "",
) -> None:
    if not hasattr(obs, "right_shoulder_rgb"):
        return

    if obs.right_shoulder_rgb is None:
        return

    if suffix:
        file_name = f"{step_idx:06d}_{suffix}.png"
    else:
        file_name = f"{step_idx:06d}.png"

    save_path = os.path.join(save_dir, file_name)
    save_rgb_image(obs.right_shoulder_rgb, save_path)
    print(f"[SERVER] Saved image -> {save_path}")


# ============================================================
# Delta action helpers
# ============================================================

def get_current_tip_pose(task_env) -> np.ndarray:
    """
    Return current end-effector tip pose:
        [x, y, z, qx, qy, qz, qw]
    """
    return np.array(
        task_env._scene.robot.arm.get_tip().get_pose(),
        dtype=np.float32,
    )


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q)

    if norm < 1e-8:
        raise ValueError("Quaternion norm too small.")

    return q / norm


def clip_delta_action(
    action: np.ndarray,
    max_pos: float = 0.03,
    max_rot_deg: float = 20.0,
    max_gripper_delta: float = 1.0,
) -> np.ndarray:
    """
    Input action:
        [dx, dy, dz, droll, dpitch, dyaw, gripper]

    The first 6 dimensions are always treated as delta control.
    """
    action = np.asarray(action, dtype=np.float32).copy()

    max_rot = np.deg2rad(max_rot_deg)

    action[:3] = np.clip(action[:3], -max_pos, max_pos)
    action[3:6] = np.clip(action[3:6], -max_rot, max_rot)
    action[6] = np.clip(action[6], -max_gripper_delta, max_gripper_delta)

    return action


def build_absolute_action_from_delta(
    cur_tip_pose: np.ndarray,
    delta_action: np.ndarray,
    gripper_cmd: float,
) -> np.ndarray:
    """
    Args:
        cur_tip_pose:
            [x, y, z, qx, qy, qz, qw]

        delta_action:
            [dx, dy, dz, droll, dpitch, dyaw, dgripper]

    Return:
        absolute action:
            [x, y, z, qx, qy, qz, qw, gripper]
    """
    cur_tip_pose = np.asarray(cur_tip_pose, dtype=np.float32)
    delta_action = np.asarray(delta_action, dtype=np.float32)

    cur_pos = cur_tip_pose[:3]
    cur_quat = normalize_quat_xyzw(cur_tip_pose[3:7])

    target_pos = cur_pos + delta_action[:3]

    cur_rot = Rotation.from_quat(cur_quat)
    delta_rot = Rotation.from_euler("xyz", delta_action[3:6], degrees=False)

    target_rot = delta_rot * cur_rot
    target_quat = normalize_quat_xyzw(target_rot.as_quat())

    return np.concatenate(
        [target_pos, target_quat, [float(gripper_cmd)]],
        axis=0,
    ).astype(np.float32)


def decide_num_substeps(
    action: np.ndarray,
    max_pos_step: float = 0.003,
    max_rot_step_deg: float = 3.0,
) -> int:
    action = np.asarray(action, dtype=np.float32)

    max_rot_step = np.deg2rad(max_rot_step_deg)

    if max_pos_step > 0:
        n_pos = int(np.ceil(np.max(np.abs(action[:3])) / max_pos_step))
    else:
        n_pos = 1

    if max_rot_step > 0:
        n_rot = int(np.ceil(np.max(np.abs(action[3:6])) / max_rot_step))
    else:
        n_rot = 1

    return max(1, n_pos, n_rot)


def resolve_gripper_command(
    cur_gripper: float,
    gripper_input: float,
    control_mode: str = "relative",
    binary_threshold: float = 0.5,
) -> float:
    """
    Resolve gripper command.

    control_mode:
        relative:
            gripper_input is delta.

        absolute:
            gripper_input is target value.
    """
    cur_gripper = float(cur_gripper)
    gripper_input = float(gripper_input)

    if control_mode == "relative":
        target = np.clip(cur_gripper + gripper_input, 0.0, 1.0)
    elif control_mode == "absolute":
        target = np.clip(gripper_input, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown gripper control mode: {control_mode}")

    return 1.0 if target > binary_threshold else 0.0


def check_task_success(
    task_env,
    reward: float,
    terminated: bool,
) -> bool:
    """
    Prefer RLBench task.success().
    Fall back to reward > 0 if unavailable.
    """
    try:
        if hasattr(task_env, "_task") and hasattr(task_env._task, "success"):
            ret = task_env._task.success()

            if isinstance(ret, tuple):
                success = ret[0]
            else:
                success = ret

            return bool(success)

    except Exception as err:
        print(f"[SERVER][WARN] success() check failed, fallback to reward: {err}")

    return bool(reward > 0.0)


def execute_action_in_substeps(
    task_env,
    obs,
    action: np.ndarray,
    max_pos_clip: float = 0.03,
    max_rot_clip_deg: float = 20.0,
    max_pos_step: float = 0.003,
    max_rot_step_deg: float = 3.0,
    gripper_control_mode: str = "relative",
    gripper_binary_threshold: float = 0.5,
    verbose: bool = True,
):
    """
    Execute one model action by splitting it into small substeps.

    Input action:
        [dx, dy, dz, droll, dpitch, dyaw, g]

    Return:
        obs, reward, terminated, failed, success
    """
    action = np.asarray(action, dtype=np.float32)

    if action.shape[0] != 7:
        raise ValueError(f"Expected action dim = 7, got shape {action.shape}")

    action = clip_delta_action(
        action=action,
        max_pos=max_pos_clip,
        max_rot_deg=max_rot_clip_deg,
        max_gripper_delta=1.0,
    )

    cur_gripper = 1.0 if float(obs.gripper_open) > 0.5 else 0.0

    final_gripper = resolve_gripper_command(
        cur_gripper=cur_gripper,
        gripper_input=float(action[6]),
        control_mode=gripper_control_mode,
        binary_threshold=gripper_binary_threshold,
    )

    num_substeps = decide_num_substeps(
        action=action,
        max_pos_step=max_pos_step,
        max_rot_step_deg=max_rot_step_deg,
    )

    sub_action = action.copy()
    sub_action[:6] /= num_substeps
    sub_action[6] = 0.0

    reward = 0.0
    terminated = False

    if verbose:
        print(f"[SERVER] Raw clipped action: {action}")
        print(f"[SERVER] Gripper control mode: {gripper_control_mode}")
        print(f"[SERVER] Current gripper: {cur_gripper}, final gripper: {final_gripper}")
        print(f"[SERVER] Split into {num_substeps} substeps")

    for i in range(num_substeps):
        cur_tip = get_current_tip_pose(task_env)
        cur_gripper_now = 1.0 if float(obs.gripper_open) > 0.5 else 0.0

        if i == num_substeps - 1:
            gripper_cmd = final_gripper
        else:
            gripper_cmd = cur_gripper_now

        action_world = build_absolute_action_from_delta(
            cur_tip_pose=cur_tip,
            delta_action=sub_action,
            gripper_cmd=gripper_cmd,
        )

        pos_delta = np.linalg.norm(action_world[:3] - cur_tip[:3])
        rot_delta = quat_distance_deg(action_world[3:7], cur_tip[3:7])

        if verbose:
            print(
                f"[SERVER] Substep {i + 1}/{num_substeps}, "
                f"pos_delta={pos_delta * 1000:.2f} mm, "
                f"rot_delta={rot_delta:.2f} deg"
            )
            print(f"[SERVER] Current tip pose: {cur_tip}")
            print(f"[SERVER] Target action_world: {action_world}")

        try:
            obs, reward, terminated = task_env.step(action_world)
        except InvalidActionError as err:
            print(f"[SERVER][ERROR] Substep {i + 1}/{num_substeps} failed: {err}")
            return obs, reward, terminated, True, False

        success = check_task_success(task_env, reward, terminated)

        if verbose:
            print(
                f"[SERVER] After substep {i + 1}/{num_substeps}: "
                f"reward={reward:.3f}, terminated={terminated}, success={success}"
            )

        if success:
            return obs, reward, True, False, True

        if terminated:
            return obs, reward, terminated, False, False

    success = check_task_success(task_env, reward, terminated)
    return obs, reward, terminated, False, success


def build_arm_action_mode(name: str):
    name = name.lower()

    if name == "ik":
        return EndEffectorPoseViaIK()

    if name == "planning":
        return EndEffectorPoseViaPlanning()

    raise ValueError(f"Unknown arm mode: {name}")


# ============================================================
# Reset helper
# ============================================================

def reset_task_with_fixed_scene(
    task_env,
    variation: int,
    rng_state,
    cam_json: str,
):
    """
    Reset task with fixed variation and optional fixed RNG state.
    Camera extrinsics are applied after reset.
    Then we fetch a fresh observation to make sure image uses fixed camera.
    """
    task_env.set_variation(int(variation))

    if rng_state is not None:
        np.random.set_state(rng_state)

    _, obs = task_env.reset()

    if cam_json is not None:
        apply_fixed_camera(cam_json)

    obs = task_env.get_observation()
    return obs


# ============================================================
# Main server
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        "RLBench server - step-by-step control via socket"
    )

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)

    parser.add_argument(
        "--episode_dir",
        type=str,
        default=None,
        help=(
            "Optional path under a collected pair. "
            "Can be pair_root, episodes/episode0, robot dir, or episode.hdf5."
        ),
    )

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--variation", type=int, default=None)
    parser.add_argument("--robot", type=str, default="panda")

    parser.add_argument("--renderer", choices=["opengl", "opengl3"], default="opengl")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--image_size", nargs=2, type=int, default=[640, 480])
    parser.add_argument("--static_positions", action="store_true")
    parser.add_argument("--dt", type=float, default=0.05)

    parser.add_argument(
        "--cam_json",
        type=str,
        default=DEFAULT_CAMERA_JSON,
    )

    parser.add_argument(
        "--save_image_dir",
        type=str,
        default="./saved_steps",
    )
    parser.add_argument("--save_init_image", action="store_true")

    parser.add_argument(
        "--arm_mode",
        type=str,
        default="ik",
        choices=["ik", "planning"],
    )

    parser.add_argument("--max_pos_clip", type=float, default=0.03)
    parser.add_argument("--max_rot_clip_deg", type=float, default=20.0)
    parser.add_argument("--max_pos_step", type=float, default=0.003)
    parser.add_argument("--max_rot_step_deg", type=float, default=3.0)

    parser.add_argument(
        "--gripper_control_mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
    )
    parser.add_argument("--gripper_binary_threshold", type=float, default=0.3)

    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def resolve_task_scene_args(args):
    """
    Resolve task, variation, and rng_state.

    Rules:
        1. If --episode_dir is provided:
            - Load rng_state.pkl from the pair folder.
            - If --task is not provided, read task from meta.json.
            - If --variation is not provided, read variation from meta.json.

        2. If --episode_dir is not provided:
            - --task must be provided.
            - --variation defaults to 0.
    """
    task_name = args.task
    variation = args.variation
    rng_state = None

    if args.episode_dir is not None:
        pair_root = find_pair_root(args.episode_dir)
        meta = load_meta(pair_root)
        rng_state = load_rng_state(pair_root)

        if task_name is None:
            task_name = meta.get("task")

        if variation is None:
            variation = meta.get("variation")

    if task_name is None:
        raise ValueError(
            "Task name is missing. Please provide --task, or use --episode_dir "
            "pointing to a pair folder that contains meta.json with field 'task'."
        )

    if variation is None:
        variation = 0

    return task_name, int(variation), rng_state


def main() -> None:
    args = parse_args()

    ensure_dir(args.save_image_dir)

    task_name, variation, rng_state = resolve_task_scene_args(args)

    print(f"[INFO] task={task_name}")
    print(f"[INFO] variation={variation}")
    print(f"[INFO] robot={args.robot}")
    print(f"[INFO] arm_mode={args.arm_mode}")
    print(f"[INFO] renderer={args.renderer}")
    print(f"[INFO] dt={args.dt}")
    print(f"[INFO] cam_json={args.cam_json}")

    if rng_state is not None:
        print("[INFO] using fixed rng_state.pkl")

    obs_cfg = build_obs_config(
        width=args.image_size[0],
        height=args.image_size[1],
        renderer=args.renderer,
    )

    env = make_env(
        robot_setup=args.robot,
        obs_config=obs_cfg,
        arm_mode=build_arm_action_mode(args.arm_mode),
        headless=args.headless,
        arm_max_velocity=1.0,
        arm_max_acceleration=4.0,
        static_positions=args.static_positions,
        dt=args.dt,
    )

    task_env = get_task_env(env, task_name)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(1)

    print(f"[SERVER] listening on {args.host}:{args.port}")

    episode_counter = 0

    try:
        while True:
            print("[SERVER] waiting for client connection...")
            conn, addr = server_sock.accept()
            print(f"[SERVER] connected by {addr}")

            with conn:
                print("[SERVER] initial reset...")

                obs = reset_task_with_fixed_scene(
                    task_env=task_env,
                    variation=variation,
                    rng_state=rng_state,
                    cam_json=args.cam_json,
                )

                cur_episode_dir = create_episode_dir(
                    args.save_image_dir,
                    episode_counter,
                )
                step_idx = 0

                if args.save_init_image:
                    save_step_image(
                        obs=obs,
                        save_dir=cur_episode_dir,
                        step_idx=step_idx,
                        suffix="reset",
                    )

                init_reply = pack_observation(
                    obs=obs,
                    reward=0.0,
                    terminated=False,
                )
                init_reply["failed"] = False
                init_reply["success"] = False

                send_msg(conn, pickle.dumps(init_reply))
                print("[SERVER] initial observation sent.")

                while True:
                    raw_data = recv_msg(conn)

                    if raw_data is None:
                        print("[SERVER] client disconnected.")
                        break

                    try:
                        msg = pickle.loads(raw_data)
                    except Exception as err:
                        print(f"[SERVER] failed to unpickle message: {err}")
                        break

                    if isinstance(msg, dict) and msg.get("command") == "reset":
                        variation_msg = msg.get("variation", variation)
                        rng_state_msg = msg.get("rng_state", rng_state)

                        obs = reset_task_with_fixed_scene(
                            task_env=task_env,
                            variation=int(variation_msg),
                            rng_state=rng_state_msg,
                            cam_json=args.cam_json,
                        )

                        episode_counter += 1
                        cur_episode_dir = create_episode_dir(
                            args.save_image_dir,
                            episode_counter,
                        )
                        step_idx = 0

                        if args.save_init_image:
                            save_step_image(
                                obs=obs,
                                save_dir=cur_episode_dir,
                                step_idx=step_idx,
                                suffix="reset",
                            )

                        reply = pack_observation(
                            obs=obs,
                            reward=0.0,
                            terminated=False,
                        )
                        reply["failed"] = False
                        reply["success"] = False

                        send_msg(conn, pickle.dumps(reply))
                        print("[SERVER] reset observation sent.")

                    elif isinstance(msg, (list, tuple, np.ndarray)):
                        action = np.asarray(msg, dtype=np.float32)

                        obs, reward, terminated, failed, success = execute_action_in_substeps(
                            task_env=task_env,
                            obs=obs,
                            action=action,
                            max_pos_clip=args.max_pos_clip,
                            max_rot_clip_deg=args.max_rot_clip_deg,
                            max_pos_step=args.max_pos_step,
                            max_rot_step_deg=args.max_rot_step_deg,
                            gripper_control_mode=args.gripper_control_mode,
                            gripper_binary_threshold=args.gripper_binary_threshold,
                            verbose=not args.quiet,
                        )

                        step_idx += 1

                        save_step_image(
                            obs=obs,
                            save_dir=cur_episode_dir,
                            step_idx=step_idx,
                        )

                        reply = pack_observation(
                            obs=obs,
                            reward=reward,
                            terminated=terminated,
                        )
                        reply["failed"] = bool(failed)
                        reply["success"] = bool(success)

                        send_msg(conn, pickle.dumps(reply))

                        print(
                            f"[SERVER] step executed, "
                            f"reward={reward:.3f}, "
                            f"terminated={terminated}, "
                            f"failed={failed}, "
                            f"success={success}"
                        )

                        if success:
                            print("[SERVER] task succeeded. stopping server.")
                            return

                    else:
                        print(f"[SERVER] unknown message type: {type(msg)}")
                        break

            print("[SERVER] connection closed.")

    except KeyboardInterrupt:
        print("\n[SERVER] shutting down...")

    finally:
        env.shutdown()
        server_sock.close()


if __name__ == "__main__":
    main()
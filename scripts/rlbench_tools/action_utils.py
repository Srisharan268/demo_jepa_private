import random
import numpy as np


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def obs_to_ee_action(obs) -> np.ndarray:
    """
    Convert RLBench observation to EE-pose action:
    [x, y, z, qx, qy, qz, qw, gripper_open]
    """
    pos = obs.gripper_pose[:3].astype(np.float32)
    quat = obs.gripper_pose[3:].astype(np.float32)

    gripper_open = getattr(obs, "gripper_open", None)
    gripper = np.array(
        [1.0 if gripper_open is None or gripper_open > 0.5 else 0.0],
        dtype=np.float32,
    )

    return np.concatenate([pos, quat, gripper], axis=0)


def demo_to_ee_actions(demo) -> np.ndarray:
    return np.stack([obs_to_ee_action(obs) for obs in demo], axis=0)


def quat_distance_deg(q_target: np.ndarray, q_current: np.ndarray) -> float:
    q_target = q_target / np.linalg.norm(q_target)
    q_current = q_current / np.linalg.norm(q_current)

    dot = abs(float(np.dot(q_target, q_current)))
    dot = np.clip(dot, -1.0, 1.0)

    return float(np.degrees(2.0 * np.arccos(dot)))
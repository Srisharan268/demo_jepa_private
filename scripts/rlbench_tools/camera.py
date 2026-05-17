import json
import math
from typing import Dict, List, Tuple

import numpy as np
from pyrep.objects.vision_sensor import VisionSensor


# ============================================================
# Quaternion / RPY conversion
# ============================================================

def rpy_to_quat_xyzw(
    roll: float,
    pitch: float,
    yaw: float,
) -> Tuple[float, float, float, float]:
    """
    Convert roll-pitch-yaw to quaternion in RLBench/PyRep order:
    [qx, qy, qz, qw].
    """
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw


def quat_xyzw_to_rpy(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> Tuple[float, float, float]:
    """
    Convert quaternion [qx, qy, qz, qw] to roll-pitch-yaw.
    """
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# ============================================================
# Camera pose getter / setter
# ============================================================

def get_cam_pose(cam_name: str) -> np.ndarray:
    """
    Get camera pose as [x, y, z, qx, qy, qz, qw].
    Must be called after env.launch().
    """
    cam = VisionSensor(cam_name)
    return np.asarray(cam.get_pose(), dtype=np.float64)


def set_cam_pose(cam_name: str, pose7: np.ndarray) -> None:
    """
    Set camera pose with [x, y, z, qx, qy, qz, qw].
    Must be called after env.launch().
    """
    pose7 = np.asarray(pose7, dtype=np.float64)
    if pose7.shape != (7,):
        raise ValueError(f"pose7 should have shape (7,), got {pose7.shape}")

    cam = VisionSensor(cam_name)
    cam.set_pose(pose7.tolist())


def get_cam_xyzrpy(cam_name: str) -> Dict[str, List[float]]:
    """
    Get camera pose in both xyz/rpy and xyz/quaternion forms.
    """
    pose = get_cam_pose(cam_name)

    x, y, z = pose[:3]
    qx, qy, qz, qw = pose[3:7]
    roll, pitch, yaw = quat_xyzw_to_rpy(qx, qy, qz, qw)

    return {
        "xyz": [float(x), float(y), float(z)],
        "rpy": [float(roll), float(pitch), float(yaw)],
        "quat_xyzw": [float(qx), float(qy), float(qz), float(qw)],
    }


def set_cam_xyzrpy(
    cam_name: str,
    xyz: Tuple[float, float, float],
    rpy: Tuple[float, float, float],
) -> None:
    """
    Set camera pose using xyz and roll-pitch-yaw.
    """
    qx, qy, qz, qw = rpy_to_quat_xyzw(rpy[0], rpy[1], rpy[2])

    pose = np.array(
        [xyz[0], xyz[1], xyz[2], qx, qy, qz, qw],
        dtype=np.float64,
    )

    set_cam_pose(cam_name, pose)


# ============================================================
# Dump / apply camera extrinsics
# ============================================================

def dump_cam_extrinsics_to_json(
    json_path: str,
    cam_names: List[str],
    strict: bool = False,
) -> None:
    """
    Dump camera extrinsics to a json file.

    Saved format:
    {
        "cam_name": {
            "xyz": [...],
            "rpy": [...]
        }
    }
    """
    data: Dict[str, Dict[str, List[float]]] = {}

    for cam_name in cam_names:
        try:
            cam_info = get_cam_xyzrpy(cam_name)
            data[cam_name] = {
                "xyz": cam_info["xyz"],
                "rpy": cam_info["rpy"],
            }
        except Exception as err:
            msg = f"[cams] failed to dump {cam_name}: {err}"
            if strict:
                raise RuntimeError(msg) from err
            print(msg)
            data[cam_name] = {"error": str(err)}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[cams] dumped camera extrinsics to {json_path}")


def apply_cam_extrinsics_from_json(
    json_path: str,
    strict: bool = False,
    verbose: bool = True,
) -> None:
    """
    Apply camera extrinsics from a json file.

    Expected format:
    {
        "cam_name": {
            "xyz": [...],
            "rpy": [...]
        }
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    for cam_name, cfg in spec.items():
        if cam_name.endswith("_mask"):
            continue

        if "error" in cfg:
            msg = f"[cams] skip {cam_name}: stored error = {cfg['error']}"
            if strict:
                raise RuntimeError(msg)
            if verbose:
                print(msg)
            continue

        try:
            xyz = tuple(float(v) for v in cfg["xyz"])
            rpy = tuple(float(v) for v in cfg["rpy"])

            if len(xyz) != 3 or len(rpy) != 3:
                raise ValueError(f"invalid xyz/rpy length: xyz={xyz}, rpy={rpy}")

            set_cam_xyzrpy(cam_name, xyz, rpy)

            if verbose:
                print(f"[cams] set {cam_name} -> xyz={list(xyz)}, rpy={list(rpy)}")

        except Exception as err:
            msg = f"[cams] skip {cam_name}: {err}"
            if strict:
                raise RuntimeError(msg) from err
            if verbose:
                print(msg)


def apply_fixed_camera(
    json_path: str,
    strict: bool = False,
    verbose: bool = False,
) -> None:
    """
    Apply fixed camera extrinsics after each scene reset.
    This is the function used by retarget.py.
    """
    apply_cam_extrinsics_from_json(
        json_path=json_path,
        strict=strict,
        verbose=verbose,
    )


# ============================================================
# Look-at RPY helper
# ============================================================

def get_look_at_rpy(
    cam_xyz: Tuple[float, float, float],
    target_xyz: Tuple[float, float, float],
    world_up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    forward_sign: int = +1,
) -> Tuple[float, float, float]:
    """
    Compute RPY for a camera located at cam_xyz and looking at target_xyz.

    forward_sign:
        +1 or -1, depending on the camera forward axis convention.
        If the rendered view is facing backward, switch this sign.
    """
    cam = np.asarray(cam_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)

    forward = target - cam
    forward_norm = np.linalg.norm(forward)

    if forward_norm < 1e-9:
        raise ValueError("camera and target are nearly at the same position")

    z_axis = forward_sign * forward / forward_norm

    up_norm = up / (np.linalg.norm(up) + 1e-12)

    if abs(np.dot(z_axis, up_norm)) > 0.999:
        up_norm = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = np.cross(up_norm, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12

    y_axis = np.cross(z_axis, x_axis)

    rot = np.column_stack((x_axis, y_axis, z_axis))

    pitch = math.asin(-rot[2, 0])
    roll = math.atan2(rot[2, 1], rot[2, 2])
    yaw = math.atan2(rot[1, 0], rot[0, 0])

    return roll, pitch, yaw
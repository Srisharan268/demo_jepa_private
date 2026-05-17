import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image

from rlbench.backend.const import VARIATION_DESCRIPTIONS


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _resize_rgb_array(
    rgb: np.ndarray,
    image_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    """
    Resize RGB image to image_hw=(H, W).
    If image_hw is None, keep original size.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)

    if image_hw is None:
        return rgb

    h, w = image_hw
    img = Image.fromarray(rgb).convert("RGB")
    img = img.resize((w, h), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _get_camera_rgb(obs, camera_name: str):
    """
    camera_name examples:
        right_shoulder_rgb
        left_shoulder_rgb
        front_rgb
        wrist_rgb
    """
    return getattr(obs, camera_name, None)


def _collect_images_from_obs(
    demo_obs: List,
    camera_names: List[str],
    image_hw: Optional[Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    """
    Collect RGB frames from RLBench observations.
    Return:
        {
            camera_name: np.ndarray with shape (T, H, W, 3)
        }
    """
    images_dict = {}

    for camera_name in camera_names:
        frames = []

        for obs in demo_obs:
            rgb = _get_camera_rgb(obs, camera_name)
            if rgb is None:
                raise ValueError(f"Observation does not contain camera field: {camera_name}")

            frames.append(_resize_rgb_array(rgb, image_hw))

        images_dict[camera_name] = np.stack(frames, axis=0).astype(np.uint8)

    return images_dict


def build_qpos_qvel_action(
    actions: np.ndarray,
    T: int,
):
    """
    Keep the same convention as the old raw2h5.py:

    qpos:
        current EE-pose target, shape (T, 8)

    qvel:
        zero velocity placeholder, shape (T, 8)

    action:
        next-step EE-pose target, shape (T, 8)
        action[t] = qpos[t + 1]
        action[-1] = qpos[-1]
    """
    actions = np.asarray(actions, dtype=np.float32)

    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(f"actions should have shape (T, 8), got {actions.shape}")

    if T <= 0:
        raise ValueError(f"T should be positive, got {T}")

    if T > actions.shape[0]:
        raise ValueError(f"T={T} is larger than actions length={actions.shape[0]}")

    qpos = actions[:T].astype(np.float32)
    qvel = np.zeros_like(qpos, dtype=np.float32)

    action = np.empty_like(qpos, dtype=np.float32)
    action[:-1] = qpos[1:]
    action[-1] = qpos[-1]

    return qpos, qvel, action


def save_demo_h5(
    out_h5: str,
    demo_obs: List,
    actions: np.ndarray,
    image_hw: Optional[Tuple[int, int]] = None,
    camera_names: Optional[List[str]] = None,
    sim: bool = True,
    attrs: Optional[Dict] = None,
) -> None:
    """
    Save one robot trajectory directly into HDF5.

    HDF5 format:
        observations/qpos
        observations/qvel
        observations/images/right_shoulder_rgb
        action

    This matches the old raw2h5.py convention.
    """
    if camera_names is None:
        camera_names = ["right_shoulder_rgb"]

    ensure_dir(os.path.dirname(out_h5))

    actions = np.asarray(actions, dtype=np.float32)

    T_obs = len(demo_obs)
    T_action = actions.shape[0]
    T = min(T_obs, T_action)

    if T <= 0:
        raise ValueError(
            f"Empty trajectory: len(demo_obs)={T_obs}, actions.shape={actions.shape}"
        )

    demo_obs = demo_obs[:T]

    qpos, qvel, action = build_qpos_qvel_action(actions, T)

    images_dict = _collect_images_from_obs(
        demo_obs=demo_obs,
        camera_names=camera_names,
        image_hw=image_hw,
    )

    with h5py.File(out_h5, "w") as root:
        root.attrs["sim"] = bool(sim)
        root.attrs["T"] = int(T)
        root.attrs["data_source"] = "direct_data_collection"

        if attrs is not None:
            for key, value in attrs.items():
                if isinstance(value, (dict, list, tuple)):
                    root.attrs[key] = json.dumps(value, ensure_ascii=False)
                else:
                    root.attrs[key] = value

        obs_group = root.create_group("observations")
        obs_group.create_dataset("qpos", data=qpos, dtype="float32")
        obs_group.create_dataset("qvel", data=qvel, dtype="float32")

        img_group = obs_group.create_group("images")
        for camera_name, imgs in images_dict.items():
            img_group.create_dataset(
                camera_name,
                data=imgs,
                dtype="uint8",
                compression="gzip",
                compression_opts=4,
            )

        root.create_dataset("action", data=action, dtype="float32")


def save_meta(pair_root: str, meta: Dict) -> None:
    ensure_dir(pair_root)

    with open(os.path.join(pair_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_rng_state(pair_root: str, rng_state) -> None:
    ensure_dir(pair_root)

    with open(os.path.join(pair_root, "rng_state.pkl"), "wb") as f:
        pickle.dump(rng_state, f)


def save_variation_descriptions(pair_root: str, descriptions) -> None:
    ensure_dir(pair_root)

    with open(os.path.join(pair_root, VARIATION_DESCRIPTIONS), "wb") as f:
        pickle.dump(descriptions, f)
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from logging import getLogger
from math import ceil
import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation


_GLOBAL_SEED = 0
logger = getLogger()

_DREAMER_PREDICTOR_PAIR_PRESETS = {
    "sim": {
        "primary_subdir": "franka",
        "reference_subdir": "sawyer",
    },
    "real": {
        "primary_subdir": "franka",
        "reference_subdir": "ur",
    },
    "sim2real": {
        "primary_subdir": "real",
        "reference_subdir": "sim",
    },
}


def init_data(
    dataset=None,
    batch_size=2,
    rank=0,
    world_size=1,
    camera_views=["right_shoulder_rgb"],
    drop_last=True,
    num_workers=12,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    data_type="real",
    camera_frame=False,
    random_erase_clip=False,
    variance_step=0,
):
    """Training loader for dreamer predictor paired-frame sampling.

    Pass root path as ``dataset=`` (preferred) or ``data_path=`` (alias).

    ``data_type``: ``sim`` | ``real`` | ``sim2real`` — see ``_DREAMER_PREDICTOR_PAIR_PRESETS``.

    ``data_fps``: optional override of HDF5 source rate used for jitter when ``reference_jitter`` is ``"fstp"``
    and for presets that default to 5 Hz. For ``sim``, default is always 30 Hz and yaml ``data_fps`` is ignored.

    ``variance_step``: optional override of variance step for reference frame selection.
    """


    if data_type not in _DREAMER_PREDICTOR_PAIR_PRESETS:
        raise ValueError(
            f"Invalid data_type={data_type!r}; expected one of {list(_DREAMER_PREDICTOR_PAIR_PRESETS)}"
        )
    preset = _DREAMER_PREDICTOR_PAIR_PRESETS[data_type]

    dataset = UnifiedDreamerPredictorPairDataset(
        dataset=dataset,
        camera_views=camera_views,
        transform=transform,
        camera_frame=camera_frame,
        random_erase_clip=random_erase_clip,
        primary_subdir=preset["primary_subdir"],
        reference_subdir=preset["reference_subdir"],
        variance_step=variance_step,
    )
    logger.info(
        "dreamer_predictor init_data: data_type=%s primary=%s reference=%s variance_step=%r",
        data_type,
        preset["primary_subdir"],
        preset["reference_subdir"],
        variance_step,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("PairedDataset data loader created")

    return data_loader, dist_sampler


class UnifiedDreamerPredictorPairDataset(torch.utils.data.Dataset):
    """Single paired-HDF5 loader for sim / real / sim2real predictor training.

    Builds ``(primary_curr, primary_target, reference_curr, reference_target)`` tuples from aligned
    primary/reference episode files under each task subdirectory. Sampling matches the legacy
    :class:`TestRLBenchSimDataset` / :class:`RealDataset` / :class:`Sim2RealDataset` behavior:

    ``reference_jitter``:
      ``"fstp"`` — jitter half-width ``ceil(episode_fps / fps)`` (former sim/franka+sawyer);

      integer ``k`` — jitter half-width ``k`` frames (former real and sim2real used ``2``).
    """

    def __init__(
        self,
        dataset,
        camera_views=("right_shoulder_rgb",),
        transform=None,
        camera_frame=False,
        random_erase_clip=False,
        primary_subdir="franka",
        reference_subdir="sawyer",
        variance_step=0,
    ):
        self.dataset = dataset
        self.camera_views = camera_views if isinstance(camera_views, (list, tuple)) else [camera_views]
        self.transform = transform
        self.camera_frame = camera_frame
        self.random_erase_clip = random_erase_clip
        self.variance_step = variance_step

        self.all_paired_episodes = []
        for task in os.listdir(dataset):
            primary_dir = os.path.join(dataset, task, primary_subdir)
            reference_dir = os.path.join(dataset, task, reference_subdir)
            primary_eps = sorted(os.listdir(primary_dir))
            reference_eps = sorted(os.listdir(reference_dir))
            if len(primary_eps) != len(reference_eps):
                shared = min(len(primary_eps), len(reference_eps))
                primary_eps = primary_eps[:shared]
                reference_eps = reference_eps[:shared]

            primary_eps = [os.path.join(primary_dir, episode) for episode in primary_eps]
            reference_eps = [os.path.join(reference_dir, episode) for episode in reference_eps]
            self.all_paired_episodes.extend(list(zip(primary_eps, reference_eps)))

        if not self.all_paired_episodes:
            raise ValueError(f"No paired episodes found under {dataset=!r}")

    def __getitem__(self, index):
        sampled_pair = self.all_paired_episodes[index]
        primary_ep = sampled_pair[0]
        reference_ep = sampled_pair[1]
        camera_view = self.camera_views[0]

        loaded_data = False
        while not loaded_data:
            try:
                with h5py.File(primary_ep, "r") as f:
                    episode_len = len(f["observations/qpos"])
                if episode_len < 2:
                    raise ValueError(f"Episode length {episode_len} is less than 2 (min for sample)")
                max_idx = episode_len - 1
                if max_idx == 0:
                    current_idx = 0
                    target_idx = 0
                else:
                    current_idx = np.random.randint(0, max_idx)
                    target_idx = np.random.randint(current_idx + 1, episode_len)

                ref_lo = max(0, current_idx - self.variance_step)
                ref_hi = min(current_idx + self.variance_step, target_idx)

                primary_indices = np.array([current_idx, target_idx], dtype=np.int64)
                if ref_hi <= ref_lo:
                    reference_current_idx = ref_lo
                else:
                    reference_current_idx = np.random.randint(ref_lo, ref_hi)
                reference_indices = np.array([reference_current_idx, target_idx], dtype=np.int64)

                primary_data = self.load_data(primary_ep, camera_view, primary_indices)
                reference_data = self.load_data(reference_ep, camera_view, reference_indices)
                loaded_data = True
            except Exception as e:
                logger.info(f"Encountered exception when loading video {e}")
                loaded_data = False
                index = np.random.randint(self.__len__())
                sampled_pair = self.all_paired_episodes[index]
                primary_ep = sampled_pair[0]
                reference_ep = sampled_pair[1]

        primary_images, _ = primary_data
        reference_images, _ = reference_data

        primary_curr = np.repeat(primary_images[:1], repeats=2, axis=0)
        primary_target = np.repeat(primary_images[1:], repeats=2, axis=0)
        reference_curr = np.repeat(reference_images[:1], repeats=2, axis=0)
        reference_target = np.repeat(reference_images[1:], repeats=2, axis=0)

        if self.transform is not None:
            primary_curr = self.transform(primary_curr)
            primary_target = self.transform(primary_target)
            reference_curr = self.transform(reference_curr)
            reference_target = self.transform(reference_target)

        return primary_curr, primary_target, reference_curr, reference_target

    def __len__(self):
        return len(self.all_paired_episodes)

    def load_data(self, episode_path, camera_view, indices):
        """Batch-read HDF5 rows preserving index order."""
        indices = np.asarray(indices, dtype=np.int64)
        uniq, inv = np.unique(indices, return_inverse=True)
        with h5py.File(episode_path, "r") as f:
            images = np.asarray(f[f"observations/images/{camera_view}"][uniq])
            states = np.asarray(f["observations/qpos"][uniq])
        images = images[inv]
        states = states[inv]
        return images, states

    def quaternion_to_euler(self, poses):
        if poses.shape[-1] == 7:
            return poses
        xyz = poses[:, :3]
        quaternions = poses[:, 3:7]
        gripper = poses[:, -1:]
        matrices = [Rotation.from_quat(quat).as_matrix() for quat in quaternions]
        euler = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in matrices]
        euler = np.stack([d for d in euler], axis=0)
        return np.concatenate([xyz, euler, gripper], axis=1)

    def poses_to_diffs(self, poses):
        xyz = poses[:, :3]
        thetas = poses[:, 3:6]
        matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
        xyz_diff = xyz[1:] - xyz[:-1]
        angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
        angle_diff = np.stack([d for d in angle_diff], axis=0)
        closedness = poses[:, -1:]
        return np.concatenate([xyz_diff, angle_diff, closedness[1:]], axis=1)

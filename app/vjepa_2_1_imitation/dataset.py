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

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation
from datetime import datetime
import pickle
import cv2

_GLOBAL_SEED = 0
logger = getLogger()

# Presets for UnifiedPairedH5Dataset (same layout as dreamer_predictor: task subdirs + HDF5 keys).
_PAIRED_H5_DOMAIN_PRESETS = {
    "sim": {"primary_subdir": "franka", "reference_subdir": "sawyer"},
    "real": {"primary_subdir": "franka", "reference_subdir": "ur"},
    "sim2real": {"primary_subdir": "real", "reference_subdir": "sim"},
}


def init_data(
    dataset,
    batch_size,
    frames_per_clip=16,
    fps=5,
    data_fps=30,
    rank=0,
    world_size=1,
    camera_views=0,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
    data_type="sim",
):
    """Training loader. Uses :class:`UnifiedPairedH5Dataset` for all ``data_type`` values.

    ``data_type``: ``sim`` | ``real`` | ``sim2real`` (see ``_PAIRED_H5_DOMAIN_PRESETS``).

    ``data_fps``: optional override of the HDF5 source rate used in ``ceil(data_fps / fps)``.
        For ``sim``, the original loader always used 30 Hz; that is fixed here and ``data_fps`` is ignored.
    """
    if data_type not in _PAIRED_H5_DOMAIN_PRESETS:
        raise ValueError(
            f"Invalid data_type={data_type!r}; expected one of {list(_PAIRED_H5_DOMAIN_PRESETS)}"
        )
    preset = _PAIRED_H5_DOMAIN_PRESETS[data_type]

    dataset = UnifiedPairedH5Dataset(
        dataset=dataset,
        camera_views=camera_views,
        frameskip=tubelet_size,
        frames_per_clip=frames_per_clip,
        fps=fps,
        data_fps=data_fps,
        transform=transform,
        camera_frame=camera_frame,
        primary_subdir=preset["primary_subdir"],
        reference_subdir=preset["reference_subdir"],
    )
    logger.info(
        "vjepa_2_1_imitation init_data: data_type=%s data_fps=%s primary=%s reference=%s",
        data_type,
        data_fps,
        preset["primary_subdir"],
        preset["reference_subdir"],
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

    logger.info("VideoDataset unsupervised data loader created")

    return data_loader, dist_sampler


class UnifiedPairedH5Dataset(torch.utils.data.Dataset):
    """Single paired-HDF5 loader for sim / real / sim2real: only subdirs and source FPS differ.

    Primary/reference HDF5 paths use ``observations/images/<camera>`` and ``observations/qpos``.
    """

    def __init__(
        self,
        dataset,
        camera_views=["left_shoulder_rgb"],
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        data_fps=30,
        transform=None,
        camera_frame=False,
        primary_subdir="franka",
        reference_subdir="sawyer",
    ):
        torch.utils.data.Dataset.__init__(self)
        self.dataset = dataset
        self.camera_views = camera_views
        self.frameskip = frameskip
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.data_fps = data_fps
        self.transform = transform
        self.camera_frame = camera_frame
        self.all_paired_episodes = []
        self.delta = False
        for task in os.listdir(dataset):
            primary_dir = os.path.join(dataset, task, primary_subdir)
            reference_dir = os.path.join(dataset, task, reference_subdir)
            primary_episodes = sorted(os.listdir(primary_dir))
            reference_episodes = sorted(os.listdir(reference_dir))
            if len(primary_episodes) != len(reference_episodes):
                shared_episodes = min(len(primary_episodes), len(reference_episodes))
                primary_episodes = primary_episodes[:shared_episodes]
                reference_episodes = reference_episodes[:shared_episodes]

            primary_episodes = [os.path.join(primary_dir, episode) for episode in primary_episodes]
            reference_episodes = [os.path.join(reference_dir, episode) for episode in reference_episodes]
            self.all_paired_episodes.extend(list(zip(primary_episodes, reference_episodes)))

        if not self.all_paired_episodes:
            raise ValueError(f"No paired episodes found under {dataset=!r}")

    def __len__(self):
        return len(self.all_paired_episodes)

    def __getitem__(self, index):
        sampled_pair = self.all_paired_episodes[index]
        primary_episode = sampled_pair[0]
        reference_episode = sampled_pair[1]
        fstp = ceil(self.data_fps / self.fps)
        nframes = int(self.frames_per_clip * fstp)

        loaded_data = False
        while not loaded_data:
            try:
                with h5py.File(primary_episode, "r") as f:
                    episode_len = len(f["observations/qpos"])
                # Need episode_len > nframes + frameskip so randint(nframes, episode_len - frameskip) is non-empty.
                if episode_len - self.frameskip <= nframes:
                    index = np.random.randint(self.__len__())
                    sampled_pair = self.all_paired_episodes[index]
                    primary_episode = sampled_pair[0]
                    reference_episode = sampled_pair[1]
                    continue
                ef = np.random.randint(nframes, episode_len - self.frameskip)
                sf = ef - nframes
                primary_indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)
                reference_indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)
                primary_data = self.load_data(primary_episode, self.camera_views[0], primary_indices)
                reference_data = self.load_data(reference_episode, self.camera_views[0], reference_indices)
                loaded_data = True
            except Exception as e:
                raise e
                loaded_data = False
                index = np.random.randint(self.__len__())
                sampled_pair = self.all_paired_episodes[index]
                primary_episode = sampled_pair[0]
                reference_episode = sampled_pair[1]

        primary_images, primary_states = primary_data
        reference_images, _ = reference_data

        primary_states = self.quaternion_to_euler(primary_states)

        primary_states = primary_states[:: self.frameskip]
        if self.delta:
            primary_actions = self.poses_to_diffs(primary_states)
        else:
            primary_actions = primary_states[1:]

        if self.transform is not None:
            primary_images = self.transform(primary_images)
            reference_images = self.transform(reference_images)
        return primary_images, reference_images, primary_actions, primary_states

    def load_data(self, episode_path, camera_view, indices):
        indices = np.asarray(indices, dtype=np.int64)
        uniq, inv = np.unique(indices, return_inverse=True)
        with h5py.File(episode_path, "r") as f:
            images = np.asarray(f[f"observations/images/{camera_view}"][uniq])
            states = np.asarray(f["observations/qpos"][uniq])
        images = images[inv]
        states = states[inv]
        states = self.quaternion_to_euler(states)

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
        xyz = poses[:, :3]  # shape [T, 3]
        thetas = poses[:, 3:6]  # euler angles, shape [T, 3]
        matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
        xyz_diff = xyz[1:] - xyz[:-1]
        angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
        angle_diff = np.stack([d for d in angle_diff], axis=0)
        closedness = poses[:, -1:]
        return np.concatenate([xyz_diff, angle_diff, closedness[1:]], axis=1)

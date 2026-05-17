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
from scipy.spatial.transform import Rotation
from datetime import datetime
import pickle
import cv2

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    dataset,
    batch_size,
    frames_per_clip=16,
    fps=5,
    data_fps=5,
    rank=0,
    world_size=1,
    camera_views=0,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    tubelet_size=2,
):
    dataset = ActionConditionedDataset(
        dataset=dataset,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        data_fps=data_fps,
        camera_views=camera_views,
        frameskip=tubelet_size,
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

        
class ActionConditionedDataset(torch.utils.data.Dataset):
    """Flat HDF5 episodes (walk ``data_path``); embodiment-agnostic naming."""

    def __init__(self,
        dataset,
        camera_views=["camera_front"],
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        transform=None,
        data_fps=5,
        **kwargs):
        self.dataset = dataset
        self.camera_views = camera_views
        self.frameskip = frameskip
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.data_fps = data_fps
        self.transform = transform

        self.all_episodes = []
        for root, _dirs, files in os.walk(dataset, followlinks=True):
            for name in files:
                if name.endswith(".hdf5") or name.endswith(".h5"):
                    self.all_episodes.append(os.path.join(root, name))
        self.all_episodes.sort()

    def __getitem__(self, index):
        episode_path = self.all_episodes[index]
        fstp = ceil(self.data_fps / self.fps)
        nframes = int(self.frames_per_clip * fstp)

        loaded_data = False
        while not loaded_data:
            try:
                with h5py.File(episode_path, "r") as f:
                    episode_len = len(f['observations/qpos'])
                if episode_len < nframes:
                    raise Exception(f"Episode is too short {episode_path=}, {nframes=}, {episode_len=}")
                ef = np.random.randint(nframes, episode_len + 1)
                sf = ef - nframes
                frame_indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)
                images, states = self.load_data(episode_path, self.camera_views[0], frame_indices)
                loaded_data = True
            except Exception as e:
                # raise e
                logger.info(f"Encountered exception when loading data {episode_path=} {e=}")
                loaded_data = False
                index = np.random.randint(self.__len__())
                episode_path = self.all_episodes[index]
        
        if self.transform is not None:
            images = self.transform(images)
        states = self.quaternion_to_euler(states)
        states = states[::self.frameskip]
        actions = self.poses_to_diffs(states)

        return images, actions, states


    def __len__(self):
        return len(self.all_episodes)
    
    def load_data(self, episode_path, camera_view, indices):
        """Batch-read HDF5 rows while preserving the order of `indices`.

        This h5py build requires fancy-index vectors to be strictly increasing;
        callers may pass unsorted or repeated frame indices. Read unique
        indices in sorted order, then scatter back with `inverse`.
        """
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
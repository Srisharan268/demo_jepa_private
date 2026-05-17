import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import wandb

def l1(a, b):
    return torch.mean(torch.abs(a - b), dim=-1)


def round_small_elements(tensor, threshold):
    mask = torch.abs(tensor) < threshold
    new_tensor = tensor.clone()
    new_tensor[mask] = 0
    return new_tensor


def cem(
    context_frame,
    context_pose,
    goal_frame,
    world_model,
    rollout=1,
    cem_steps=100,
    momentum_mean=0.25,
    momentum_std=0.95,
    momentum_mean_gripper=0.15,
    momentum_std_gripper=0.15,
    samples=100,
    topk=10,
    verbose=False,
    maxnorm=0.05,
    axis={},
    objective=l1,
    close_gripper=None,
    use_rpy=True,
    abs_gripper=True,
    fixed_gripper=None,
):
    """
    :param context_frame: [B=1, T=1, HW, D]
    :param goal_frame: [B=1, T=1, HW, D]
    :param world_model: f(context_frame, action) -> next_frame [B, 1, HW, D]
    :return: [B=1, rollout, 7] an action trajectory over rollout horizon

    Cross-Entropy Method
    -----------------------
    1. for rollout horizon:
    1.1. sample several actions
    1.2. compute next states using WM
    3. compute similarity of final states to goal_frames
    4. select topk samples and update mean and std using topk action trajs
    5. choose final action to be mean of distribution
    """
    context_frame = context_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    goal_frame = goal_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    context_pose = context_pose.repeat(samples, 1, 1)  # Reshape to [S, 1, 7]

    mean = torch.cat(
        [
            torch.zeros((rollout, 3), device=context_frame.device),
            torch.zeros((rollout, 3), device=context_frame.device) if use_rpy else torch.zeros((rollout, 0), device=context_frame.device),
            torch.zeros((rollout, 1), device=context_frame.device),
        ],
        dim=-1,
    )

    std = torch.cat(
        [
            torch.ones((rollout, 3), device=context_frame.device) * maxnorm,
            torch.ones((rollout, 3), device=context_frame.device) * maxnorm if use_rpy else torch.zeros((rollout, 0), device=context_frame.device),
            torch.ones((rollout, 1), device=context_frame.device),
        ],
        dim=-1,
    )

    for ax in axis.keys():
        mean[:, ax] = axis[ax]

    if fixed_gripper is not None:
        mean[:, -1] = fixed_gripper
        std[:, -1] = 0.0

    def sample_action_traj():
        """Sample several action trajectories"""
        action_traj, frame_traj, pose_traj = None, context_frame, context_pose

        for h in range(rollout):

            action_samples = torch.randn(samples, mean.size(1), device=mean.device) * std[h] + mean[h]
            action_samples[:, :-1] = torch.clip(action_samples[:, :-1], min=-maxnorm, max=maxnorm)
            if abs_gripper:
                action_samples[:, -1:] = torch.clip(action_samples[:, -1:], min=0.0, max=1.0)
            else:
                action_samples[:, -1:] = torch.clip(action_samples[:, -1:], min=-maxnorm, max=maxnorm)
            for ax in axis.keys():
                action_samples[:, ax] = axis[ax]

            action_samples = torch.cat(
                [
                    action_samples[:, :-1],
                    torch.zeros((len(action_samples), 3), device=mean.device) if not use_rpy else torch.zeros((len(action_samples), 0), device=mean.device),
                    action_samples[:, -1:],
                ],
                dim=-1,
            )[:, None]
            action_samples = action_samples.to(context_frame.dtype)
            if close_gripper is not None and h >= close_gripper:
                action_samples[:, :, -1] = 1.0
            if fixed_gripper is not None:
                action_samples[:, :, -1] = fixed_gripper

            action_traj = (
                torch.cat([action_traj, action_samples], dim=1) if action_traj is not None else action_samples
            )

            next_frame, next_pose = world_model(frame_traj, action_traj, pose_traj)
            frame_traj = torch.cat([frame_traj, next_frame], dim=1)
            pose_traj = torch.cat([pose_traj, next_pose], dim=1)

        return action_traj, frame_traj

    def select_topk_action_traj(final_state, goal_state, actions):
        """Get the topk action trajectories that bring us closest to goal"""
        sims = objective(final_state.flatten(1), goal_state.flatten(1))
        indices = sims.topk(topk, largest=False).indices
        selected_actions = actions[indices]
        best_loss = sims[indices[0]].item()

        try:
            if wandb.run is not None:
                wandb.log({"best_sample_l1": best_loss})
        except Exception:
            pass

        return selected_actions, best_loss

    best_loss = float('inf')
    for step in tqdm(range(cem_steps), disable=True):
        action_traj, frame_traj = sample_action_traj()
        selected_actions, step_best_loss = select_topk_action_traj(
            final_state=frame_traj[:, -1], goal_state=goal_frame, actions=action_traj
        )
        best_loss = step_best_loss
        mean_selected_actions = selected_actions.mean(dim=0)
        std_selected_actions = selected_actions.std(dim=0)

        mean = torch.cat(
            [
                mean_selected_actions[..., :3] * (1.0 - momentum_mean) + mean[..., :3] * momentum_mean,
                mean_selected_actions[..., 3:-1] * (1.0 - momentum_mean) + mean[..., 3:-1] * momentum_mean if use_rpy else torch.zeros((mean_selected_actions.size(0), 0), device=context_frame.device),
                mean_selected_actions[..., -1:] * (1.0 - momentum_mean_gripper)
                + mean[..., -1:] * momentum_mean_gripper,
            ],
            dim=-1,
        )
        std = torch.cat(
            [
                std_selected_actions[..., :3] * (1.0 - momentum_std) + std[..., :3] * momentum_std,
                std_selected_actions[..., 3:-1] * (1.0 - momentum_std) + std[..., 3:-1] * momentum_std if use_rpy else torch.zeros((std_selected_actions.size(0), 0), device=context_frame.device),
                std_selected_actions[..., -1:] * (1.0 - momentum_std_gripper) + std[..., -1:] * momentum_std_gripper,
            ],
            dim=-1,
        )
        if fixed_gripper is not None:
            mean[:, -1] = fixed_gripper
            std[:, -1] = 0.0
        # print(mean)

    new_action = torch.cat(
        [
            mean[..., :-1],
            round_small_elements(mean[..., -1:], 0.25),
        ],
        dim=-1,
    )[None, :]

    return new_action, best_loss


def compute_new_pose(pose, action, abs_gripper=True):
    """
    Integrate one action step into pose (xyz + euler + gripper). Position and orientation use deltas.

    :param pose: [B, T=1, 7] xyz(3), euler xyz(3), gripper(1)
    :param action: [B, T=1, 7] same layout
    :param abs_gripper: if True, action[..., -1] is absolute gripper in [0, 1]. If False, it is a
        delta added to pose[..., -1] (then clipped to [0, 1]).
    :returns: [B, T=1, 7]
    """
    device, dtype = pose.device, pose.dtype
    pose = pose[:, 0].to(torch.float32).cpu().numpy()
    action = action[:, 0].to(torch.float32).cpu().numpy()
    new_xyz = pose[:, :3] + action[:, :3]
    thetas = pose[:, 3:6]
    delta_thetas = action[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas]
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    new_angle = np.stack([d for d in angle_diff], axis=0)
    if abs_gripper:
        new_closedness = np.clip(action[:, -1:], 0, 1)
    else:
        new_closedness = np.clip(pose[:, -1:] + action[:, -1:], 0, 1)
    new_pose = np.concatenate([new_xyz, new_angle, new_closedness], axis=-1)
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]


def quaternion_to_euler(poses):
    if poses.shape[-1] == 7:
        return poses
    xyz = poses[:, :3]
    quaternions = poses[:, 3:7]
    gripper = poses[:, -1:]
    matrices = [Rotation.from_quat(quat).as_matrix() for quat in quaternions]
    euler = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in matrices]
    euler = np.stack([d for d in euler], axis=0)
    return np.concatenate([xyz, euler, gripper], axis=1)

def euler_to_quaternion(poses):
    if poses.shape[-1] == 8:
        return poses
    xyz = poses[..., :3]
    euler = poses[..., 3:6]
    gripper = poses[..., 6:]
    quats = [Rotation.from_euler("xyz", angles, degrees=False).as_quat() for angles in euler]
    quats = np.stack(quats, axis=0)
    result = np.concatenate([xyz, quats, gripper], axis=-1)
    return result

def poses_to_diffs(poses):
    xyz = poses[:, :3]
    thetas = poses[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    xyz_diff = xyz[1:] - xyz[:-1]
    angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    angle_diff = np.stack([d for d in angle_diff], axis=0)
    closedness = poses[:, -1:]
    return np.concatenate([xyz_diff, angle_diff, closedness[1:]], axis=1)

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def latent_l1(a, b):
    return F.l1_loss(a.flatten(1), b.flatten(1), reduction="mean").item()


@torch.no_grad()
def calibrate_threshold_from_sequence(
    world_model,
    hdf5_path,
    img_key="observations/images/right_shoulder_rgb",
    data_fps=10,
    target_fps=5,
    max_future_steps=6,
    use_quantile=False,
    pos_quantile=0.95,
):
    frame_skip = data_fps // target_fps

    with h5py.File(hdf5_path, "r") as f:
        imgs = np.array(f[img_key])

    T = len(imgs)
    print(f"num_frames = {T}, frame_skip = {frame_skip}")

    reps = []
    for i in range(T):
        rep = world_model.encode(imgs[i])
        if world_model.normalize_reps:
            rep = F.layer_norm(rep, (rep.size(-1),))
        reps.append(rep.detach().cpu())

    pos_dists = []
    neg_dists = []

    for i in range(0, T - (max_future_steps + 1) * frame_skip):
        x_t = reps[i].to(world_model.device, dtype=world_model.dtype)
        y_t = reps[i].to(world_model.device, dtype=world_model.dtype)
        y_tp1 = reps[i + frame_skip].to(world_model.device, dtype=world_model.dtype)

        pred_x_next = world_model.forward_dreamer_predictor(x_t, y_t, y_tp1)
        pred_x_next = pred_x_next.detach().cpu()

        for k in range(1, max_future_steps + 1):
            true_rep = reps[i + k * frame_skip]
            d = latent_l1(pred_x_next, true_rep)

            if k <= 3:
                pos_dists.append(d)
            else:
                neg_dists.append(d)

    pos_dists = np.array(pos_dists, dtype=np.float32)
    neg_dists = np.array(neg_dists, dtype=np.float32)

    print(f"num_pos = {len(pos_dists)}, num_neg = {len(neg_dists)}")
    print(f"pos mean={pos_dists.mean():.6f}, median={np.median(pos_dists):.6f}")
    print(f"neg mean={neg_dists.mean():.6f}, median={np.median(neg_dists):.6f}")

    if use_quantile:
        threshold = float(np.quantile(pos_dists, pos_quantile))
        method = f"pos_{int(pos_quantile*100)}th_quantile"
    else:
        candidates = np.unique(np.concatenate([pos_dists, neg_dists]))
        best_thr, best_bacc = None, -1.0

        for thr in candidates:
            pos_acc = (pos_dists <= thr).mean()
            neg_acc = (neg_dists > thr).mean()
            bacc = 0.5 * (pos_acc + neg_acc)
            if bacc > best_bacc:
                best_bacc = bacc
                best_thr = float(thr)

        threshold = best_thr
        method = f"best_balanced_acc={best_bacc:.6f}"

    print(f"[threshold] {threshold:.6f} ({method})")

    stats = {
        "threshold": threshold,
        "method": method,
        "pos_mean": float(pos_dists.mean()),
        "pos_median": float(np.median(pos_dists)),
        "pos_p90": float(np.quantile(pos_dists, 0.90)),
        "pos_p95": float(np.quantile(pos_dists, 0.95)),
        "neg_mean": float(neg_dists.mean()),
        "neg_median": float(np.median(neg_dists)),
        "neg_p10": float(np.quantile(neg_dists, 0.10)),
        "neg_p50": float(np.quantile(neg_dists, 0.50)),
        "num_pos": int(len(pos_dists)),
        "num_neg": int(len(neg_dists)),
    }
    return threshold, stats, pos_dists, neg_dists

@torch.no_grad()
def export_frame_similarity_matrix(
    world_model,
    hdf5_path,
    out_dir,
    img_key="observations/images/right_shoulder_rgb",
    max_frames=None,
    save_csv=True,
    save_npy=True,
    save_png=True,
):
    import os
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        imgs = np.array(f[img_key])

    if max_frames is not None:
        imgs = imgs[:max_frames]

    num_frames = len(imgs)
    print(f"num_frames = {num_frames}")

    reps = []
    for i in range(num_frames):
        rep = world_model.encode(imgs[i])
        if world_model.normalize_reps:
            rep = F.layer_norm(rep, (rep.size(-1),))
        rep = rep.flatten(1).squeeze(0).detach().float().cpu()
        reps.append(rep)

    reps = torch.stack(reps, dim=0)
    print(f"reps shape = {tuple(reps.shape)}")

    dist = torch.cdist(reps, reps, p=1) / reps.shape[1]
    dist_np = dist.numpy()

    base = os.path.join(out_dir, "frame_l1_matrix")

    if save_npy:
        np.save(base + ".npy", dist_np)
        print(f"saved: {base}.npy")

    if save_csv:
        np.savetxt(base + ".csv", dist_np, delimiter=",", fmt="%.6f")
        print(f"saved: {base}.csv")

    if save_png:
        plt.figure(figsize=(8, 6))
        im = plt.imshow(dist_np)
        plt.colorbar(im)
        plt.title("Frame-to-Frame Latent L1 Distance Matrix")
        plt.xlabel("Frame Index")
        plt.ylabel("Frame Index")
        plt.tight_layout()
        plt.savefig(base + ".png", dpi=200)
        plt.close()
        print(f"saved: {base}.png")

    return dist_np

class WorldModel(object):

    def __init__(
        self,
        encoder,
        predictor,
        dreamer_predictor,
        tokens_per_frame,
        transform,
        tubelet_size=2,
        mpc_args={
            "rollout": 1,
            "samples": 200,
            "topk": 10,
            "cem_steps": 50,
            "momentum_mean": 0.15,
            "momentum_std": 0.75,
            "maxnorm": 0.1,
            "verbose": True,
        },
        normalize_reps=True,
        device="cuda:0",
        dtype=torch.bfloat16,
        abs_gripper=True,
        discrete_gripper=False,
    ):
        super().__init__()
        self.encoder = encoder.to(device)
        self.predictor = predictor.to(device)
        self.dreamer_predictor = dreamer_predictor.to(device)
        self.normalize_reps = normalize_reps
        self.transform = transform
        self.tokens_per_frame = tokens_per_frame
        self.tubelet_size = tubelet_size
        self.device = device
        self.mpc_args = mpc_args
        self.dtype = dtype
        #: If True, MPC last channel is absolute open/close in [0,1]; if False, delta on current gripper.
        self.abs_gripper = abs_gripper
        self.discrete_gripper = discrete_gripper

    def encode(self, image):
        clip = np.expand_dims(image, axis=0)
        clip = self.transform(clip)[None, :]
        B, C, T, H, W = clip.size()
        clip = (
            clip.permute(0, 2, 1, 3, 4)
            .flatten(0, 1)
            .unsqueeze(2)
            .repeat(1, 1, self.tubelet_size, 1, 1)
        )
        clip = clip.to(self.dtype).to(self.device, non_blocking=True)
        # V-JEPA 2.1: training=False returns embed_dim output
        h = self.encoder(clip, masks=None, training=False)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if self.normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h

    def forward_dreamer_predictor(self, xt, yt, yt_plus_1):
        return self.dreamer_predictor(xt=xt, yt=yt, yt_plus_1=yt_plus_1)

    def __call__(self, current_img, pose, current_ref, target_ref, target_img=None):
        current_rep = self.encode(current_img)
        current_ref_rep = self.encode(current_ref)
        target_ref_rep = self.encode(target_ref)
        goal_rep = self.forward_dreamer_predictor(current_rep, current_ref_rep, target_ref_rep)
        if target_img is not None:
            target_rep = self.encode(target_img)
            distance = F.l1_loss(target_rep.flatten(1), goal_rep.flatten(1))
            print(distance)
        pose = quaternion_to_euler(pose)
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose)
        pose = pose.to(self.dtype).to(self.device)

        def wm(reps, actions, poses):
            B, T, N_T, D = reps.size()
            reps = reps.flatten(1, 2)
            next_rep = self.predictor(reps, actions, poses)[:, -self.tokens_per_frame :]
            if self.normalize_reps:
                next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
            next_rep = next_rep.view(B, 1, N_T, D)
            next_pose = compute_new_pose(
                poses[:, -1:], actions[:, -1:], abs_gripper=self.abs_gripper
            )
            return next_rep, next_pose

        mpc_kwargs = dict(self.mpc_args)
        mpc_kwargs["abs_gripper"] = self.abs_gripper

        if self.discrete_gripper:
            action_g0, loss_g0 = cem(
                context_frame=current_rep,
                context_pose=pose,
                goal_frame=goal_rep,
                world_model=wm,
                fixed_gripper=0.0,
                **mpc_kwargs,
            )
            action_g1, loss_g1 = cem(
                context_frame=current_rep,
                context_pose=pose,
                goal_frame=goal_rep,
                world_model=wm,
                fixed_gripper=1.0,
                **mpc_kwargs,
            )
            if loss_g0 <= loss_g1:
                action = action_g0[0]
                print(f"[discrete_gripper] chose gripper=0 (loss={loss_g0:.6f} vs {loss_g1:.6f})")
            else:
                action = action_g1[0]
                print(f"[discrete_gripper] chose gripper=1 (loss={loss_g1:.6f} vs {loss_g0:.6f})")
        else:
            action, _ = cem(
                context_frame=current_rep,
                context_pose=pose,
                goal_frame=goal_rep,
                world_model=wm,
                **mpc_kwargs,
            )
            action = action[0]

        return action, goal_rep.detach()

    def dummy_test(self, current_img, pose, gt_target_img):
        current_rep = self.encode(current_img)
        gt_target_rep = self.encode(gt_target_img)
        pose = quaternion_to_euler(pose)
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose)
        pose = pose.to(self.dtype).to(self.device)

        def wm(reps, actions, poses):
            B, T, N_T, D = reps.size()
            reps = reps.flatten(1, 2)
            next_rep = self.predictor(reps, actions, poses)[:, -self.tokens_per_frame :]
            if self.normalize_reps:
                next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
            next_rep = next_rep.view(B, 1, N_T, D)
            next_pose = compute_new_pose(
                poses[:, -1:], actions[:, -1:], abs_gripper=self.abs_gripper
            )
            return next_rep, next_pose

        mpc_kwargs = dict(self.mpc_args)
        mpc_kwargs["abs_gripper"] = self.abs_gripper

        action, _ = cem(
            context_frame=current_rep,
            context_pose=pose,
            goal_frame=gt_target_rep,
            world_model=wm,
            **mpc_kwargs,
        )
        action = action[0]

        return action, gt_target_rep.detach()

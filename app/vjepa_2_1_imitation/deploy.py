#!/usr/bin/env python3
import argparse
import copy
import os
import pickle
import random
import socket
import struct
from collections import deque
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from app.vjepa_2_1_imitation.transforms import make_transforms
from app.vjepa_2_1_imitation.utils import (
    init_encoder_predictor,
    init_dreamer_predictor,
    load_pretrained_encoder_predictor,
    load_dreamer_predictor,
    vjepa_2_1_encoder_args_from_cfg,
)
from app.vjepa_2_1_imitation.diffusion_head import DiffusionHead

from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.logging import get_logger


_GLOBAL_SEED = 911

random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)

def init_socket(ip: str = "127.0.0.1", port: int = 9001) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    return sock


def send_msg(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)


def recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    data = b""

    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet

    return data


def recv_msg(sock: socket.socket) -> Optional[bytes]:
    raw_len = recvall(sock, 4)

    if not raw_len:
        return None

    msg_len = struct.unpack(">I", raw_len)[0]
    return recvall(sock, msg_len)

def make_fake_data() -> dict:
    return {
        "gripper_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "gripper_open": 1.0,
        "reward": 0.0,
        "terminated": False,
        "failed": False,
        "success": False,
        "right_shoulder_rgb": np.random.randint(
            0,
            256,
            size=(480, 640, 3),
            dtype=np.uint8,
        ),
    }


def to_1d_array(x, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)

    if arr.ndim == 0:
        return arr.reshape(1)

    return arr.reshape(-1)


def process_observation(data: dict) -> Tuple[np.ndarray, np.ndarray]:

    if data.get("gripper_pose") is None:
        raise ValueError("Server observation missing gripper_pose.")

    if data.get("gripper_open") is None:
        raise ValueError("Server observation missing gripper_open.")

    if data.get("right_shoulder_rgb") is None:
        raise ValueError("Server observation missing right_shoulder_rgb.")

    pose7 = to_1d_array(data["gripper_pose"], dtype=np.float32)
    gripper = to_1d_array(data["gripper_open"], dtype=np.float32)

    if pose7.shape[0] != 7:
        raise ValueError(f"Expected gripper_pose dim=7, got shape={pose7.shape}")

    pose8 = np.concatenate([pose7, gripper[:1]], axis=-1)
    pose8 = pose8[None, :].astype(np.float32)

    current_img = np.asarray(data["right_shoulder_rgb"], dtype=np.uint8)

    return current_img, pose8


def server_episode_finished(data: dict) -> bool:

    if bool(data.get("success", False)):
        print("[CLIENT] Server reports success. Stop rollout.")
        return True

    if bool(data.get("failed", False)):
        print("[CLIENT][WARN] Previous action failed. Continue.")

    if bool(data.get("terminated", False)):
        print("[CLIENT] Server reports terminated. Stop rollout.")
        return True

    return False

def find_first_existing_h5_key(h5_file: h5py.File, candidates) -> Optional[str]:
    for key in candidates:
        if key is not None and key in h5_file:
            return key
    return None


class ReferenceLoader:
    """
    Load a fixed reference trajectory from local HDF5.

    Server does not send reference information.
    Reference path must be provided by deploy.reference_h5 in YAML.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        data_fps: int = 30,
        target_fps: int = 5,
        image_key: Optional[str] = None,
    ):
        self.data_fps = int(data_fps)
        self.target_fps = int(target_fps)
        self.frame_skip = max(1, self.data_fps // self.target_fps)
        self.image_key = image_key

        self.path = None
        self.img_data = None
        self.length = 0
        self.current_idx = 0

        if path is not None:
            self.load(path)

    def load(self, path: str) -> None:
        path = os.path.abspath(path)

        if self.path == path and self.img_data is not None:
            self.reset()
            return

        image_key_candidates = [
            self.image_key,
            "observations/images/camera_front",
            "observations/images/right_shoulder_rgb",
            "observations/images/front_rgb",
        ]

        with h5py.File(path, "r") as f:
            final_image_key = find_first_existing_h5_key(f, image_key_candidates)

            if final_image_key is None:
                tried = [k for k in image_key_candidates if k is not None]
                raise KeyError(f"No valid image key found in {path}. Tried: {tried}")

            self.image_key = final_image_key
            self.img_data = np.asarray(f[final_image_key])
            self.length = int(self.img_data.shape[0])

        if self.length <= 1:
            raise ValueError(f"Reference episode too short: {path}, length={self.length}")

        self.path = path
        self.reset()

        print(
            f"[CLIENT] Reference loaded: {self.path}, "
            f"num_frames={self.length}, "
            f"image_key={self.image_key}, "
            f"frame_skip={self.frame_skip}"
        )

    def _get_indices(self) -> Tuple[int, int]:
        if self.img_data is None:
            raise RuntimeError("ReferenceLoader has not loaded any HDF5 file.")

        current_idx = min(self.current_idx, self.length - 1)
        target_idx = min(current_idx + self.frame_skip, self.length - 1)

        return current_idx, target_idx

    def pop_and_update(self) -> Tuple[np.ndarray, np.ndarray]:
        current_idx, target_idx = self._get_indices()

        current_img = self.img_data[current_idx]
        target_img = self.img_data[target_idx]

        if self.current_idx < self.length - 1:
            self.current_idx = min(self.current_idx + self.frame_skip, self.length - 1)

        return current_img, target_img

    def pop_without_update(self) -> Tuple[np.ndarray, np.ndarray]:
        current_idx, target_idx = self._get_indices()

        current_img = self.img_data[current_idx]
        target_img = self.img_data[target_idx]

        return current_img, target_img

    def reset(self) -> None:
        self.current_idx = 0


def resolve_reference_h5(params: dict) -> str:
    deploy_cfg = params.get("deploy", {})
    reference_h5 = deploy_cfg.get("reference_h5", None)

    if reference_h5 is None:
        raise RuntimeError(
            "No reference HDF5 path found. "
            "Please set deploy.reference_h5 in YAML."
        )

    return os.path.abspath(reference_h5)


def build_reference_loader(params: dict) -> ReferenceLoader:
    deploy_cfg = params.get("deploy", {})

    reference_h5_path = resolve_reference_h5(params)

    image_key = deploy_cfg.get("image_key", "observations/images/camera_front")
    ref_data_fps = int(deploy_cfg.get("ref_data_fps", 30))
    ref_target_fps = int(deploy_cfg.get("ref_target_fps", 5))

    ref_pool = ReferenceLoader(
        path=reference_h5_path,
        data_fps=ref_data_fps,
        target_fps=ref_target_fps,
        image_key=image_key,
    )

    print(f"[CLIENT] Fixed reference h5: {reference_h5_path}")

    return ref_pool


# ============================================================
# Latent-distance reference progress
# ============================================================

def latent_l1_distance(a, b) -> float:
    return F.l1_loss(a.flatten(1), b.flatten(1)).item()


def cal_l1_distance(goal, buffer):
    dists = [latent_l1_distance(goal, rep) for rep in buffer]
    min_idx = int(np.argmin(dists))
    min_dist = float(dists[min_idx])
    return dists, min_idx, min_dist


def select_reference_pair(
    world_model,
    current_img: np.ndarray,
    prev_goal,
    obs_buffer,
    ref_pool: ReferenceLoader,
    dtype,
    mixed_precision: bool,
    l1_threshold: float,
):
    with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
        current_rep = world_model.encode(current_img)

    current_rep = current_rep.detach()
    obs_buffer.append(current_rep)

    if prev_goal is not None and len(obs_buffer) > 0:
        _, _, l1_dist = cal_l1_distance(prev_goal, list(obs_buffer))
        print(f"[CLIENT] latent l1 dist: {l1_dist:.6f}")

        if l1_dist < l1_threshold:
            print("[CLIENT] dist < l1_threshold, advance reference")
            current_ref, target_ref = ref_pool.pop_and_update()
            obs_buffer.clear()
        else:
            print("[CLIENT] dist >= l1_threshold, retry current reference")
            current_ref, target_ref = ref_pool.pop_without_update()
    else:
        current_ref, target_ref = ref_pool.pop_without_update()

    return current_ref, target_ref


# ============================================================
# Checkpoint loading
# ============================================================

def load_diffusion_head(
    diffusion_head_path,
    diffusion_head,
    replace_kw=("backbone.", "module."),
    state_key="diffusion_head",
):
    """
    Load diffusion_head weights from a training checkpoint.
    Training checkpoint saves diffusion head under the diffusion_head key.
    """
    logger.info(f"Loading diffusion_head checkpoint from {diffusion_head_path}")

    ckpt = robust_checkpoint_loader(
        diffusion_head_path,
        map_location=torch.device("cpu"),
    )

    state_dict = ckpt[state_key]

    for kw in replace_kw:
        state_dict = {k.replace(kw, ""): v for k, v in state_dict.items()}

    msg = diffusion_head.load_state_dict(state_dict, strict=False)
    logger.info(f"loaded diffusion_head with msg: {msg}")

    del ckpt
    return diffusion_head

class WorldModel:

    def __init__(
        self,
        encoder,
        dreamer_predictor,
        diffusion_head,
        tokens_per_frame,
        transform,
        tubelet_size=2,
        normalize_reps=True,
        device="cuda:0",
        dtype=torch.bfloat16,
    ):
        self.encoder = encoder.to(device)
        self.dreamer_predictor = dreamer_predictor.to(device)
        self.diffusion_head = diffusion_head.to(device)

        self.tokens_per_frame = tokens_per_frame
        self.transform = transform
        self.tubelet_size = tubelet_size
        self.normalize_reps = normalize_reps
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
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

        h = self.encoder(clip, masks=None, training=False)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)

        if self.normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))

        return h

    @torch.no_grad()
    def forward_dreamer_predictor(self, xt, yt, yt_plus_1):
        return self.dreamer_predictor(
            xt=xt,
            yt=yt,
            yt_plus_1=yt_plus_1,
        )

    @torch.no_grad()
    def __call__(self, current_img, pose, current_ref, target_ref, target_img=None):
        current_rep = self.encode(current_img)
        current_ref_rep = self.encode(current_ref)
        target_ref_rep = self.encode(target_ref)

        dreamer_target_feature = self.forward_dreamer_predictor(
            current_rep,
            current_ref_rep,
            target_ref_rep,
        )

        if target_img is not None:
            target_rep = self.encode(target_img)
            distance = F.l1_loss(
                target_rep.flatten(1),
                dreamer_target_feature.flatten(1),
            )
            print(f"[debug] target_rep <-> dreamer_target L1 = {distance.item():.6f}")

        cond = dreamer_target_feature.reshape(
            -1,
            self.tokens_per_frame,
            dreamer_target_feature.shape[-1],
        )
        cond = cond.to(self.dtype).to(self.device)

        action = self.diffusion_head.denoise_inference(cond=cond)

        return action, dreamer_target_feature.detach()


# ============================================================
# Model construction
# ============================================================

def resolve_dtype(dtype_name: str):
    dtype_name = str(dtype_name).lower()

    if dtype_name == "bfloat16":
        return torch.bfloat16, True

    if dtype_name == "float16":
        return torch.float16, True

    return torch.float32, False


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        return device

    return torch.device("cpu")


def freeze_eval(module, dtype, device):
    for param in module.parameters():
        param.requires_grad = False

    module.to(dtype).to(device)
    module.eval()

    return module


def build_world_model(args: dict):
    cfgs_meta = args.get("meta", {})
    cfgs_model = args.get("model", {})
    cfgs_data = args.get("data", {})
    cfgs_data_aug = args.get("data_aug", {})
    cfgs_loss = args.get("loss", {})
    cfgs_diffusion = args.get("diffusion", {})

    folder = args.get("folder", None)

    resume_checkpoint = cfgs_meta.get("resume_checkpoint", None)
    pretrain_checkpoint = cfgs_meta.get("pretrain_checkpoint", None)
    dreamer_checkpoint = cfgs_meta.get("dreamer_predictor_checkpoint", None)
    diffusion_head_checkpoint = cfgs_meta.get("diffusion_head_checkpoint", None)

    if diffusion_head_checkpoint is None:
        if resume_checkpoint is not None:
            diffusion_head_checkpoint = resume_checkpoint
        elif folder is not None:
            diffusion_head_checkpoint = os.path.join(folder, "latest.pt")
        else:
            raise RuntimeError(
                "No diffusion head checkpoint found. "
                "Please set meta.diffusion_head_checkpoint or meta.resume_checkpoint."
            )

    load_predictor = cfgs_meta.get("load_predictor", True)
    load_encoder = cfgs_meta.get("load_encoder", True)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
    use_sdpa = cfgs_meta.get("use_sdpa", False)

    dtype_name = cfgs_meta.get("dtype", "bfloat16")
    dtype, mixed_precision = resolve_dtype(dtype_name)
    logger.info(f"dtype={dtype_name}, mixed_precision={mixed_precision}")

    device = get_device()

    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)

    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    use_extrinsics = cfgs_model.get("use_extrinsics", False)
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)

    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size")
    tubelet_size = cfgs_data.get("tubelet_size", 2)

    normalize_reps = cfgs_loss.get("normalize_reps", True)

    trajectory_dim = cfgs_diffusion.get("trajectory_dim")
    cond_dim = cfgs_diffusion.get("cond_dim")
    horizon = cfgs_diffusion.get("horizon")
    n_cond_steps = cfgs_diffusion.get("n_cond_steps")
    num_train_timesteps = cfgs_diffusion.get("num_train_timesteps")
    beta_start = cfgs_diffusion.get("beta_start")
    beta_end = cfgs_diffusion.get("beta_end")
    beta_schedule = cfgs_diffusion.get("beta_schedule")
    variance_type = cfgs_diffusion.get("variance_type")
    clip_sample = cfgs_diffusion.get("clip_sample")
    prediction_type = cfgs_diffusion.get("prediction_type")
    n_layer = cfgs_diffusion.get("n_layer")
    n_cond_layers = cfgs_diffusion.get("n_cond_layers")
    n_head = cfgs_diffusion.get("n_head")
    n_emb = cfgs_diffusion.get("n_emb")
    p_drop_emb = cfgs_diffusion.get("p_drop_emb")
    p_drop_attn = cfgs_diffusion.get("p_drop_attn")
    causal_attn = cfgs_diffusion.get("causal_attn")
    time_as_cond = cfgs_diffusion.get("time_as_cond")

    encoder, _ = init_encoder_predictor(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=512,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        action_embed_dim=7,
        pred_is_frame_causal=pred_is_frame_causal,
        use_extrinsics=use_extrinsics,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        **vjepa_2_1_encoder_args_from_cfg(cfgs_model),
    )

    dreamer_predictor = init_dreamer_predictor(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )

    diffusion_head = DiffusionHead(
        trajectory_dim=trajectory_dim,
        cond_dim=cond_dim,
        horizon=horizon,
        n_cond_steps=n_cond_steps,
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        variance_type=variance_type,
        clip_sample=clip_sample,
        prediction_type=prediction_type,
        n_layer=n_layer,
        n_cond_layers=n_cond_layers,
        n_head=n_head,
        n_emb=n_emb,
        p_drop_emb=p_drop_emb,
        p_drop_attn=p_drop_attn,
        causal_attn=causal_attn,
        time_as_cond=time_as_cond,
    ).to(device)

    if compile_model:
        logger.info("Compiling encoder, dreamer_predictor, and diffusion_head.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        dreamer_predictor.compile()
        diffusion_head.compile()

    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    logger.info("Loading pretrained encoder weights...")
    encoder, _, _ = load_pretrained_encoder_predictor(
        r_path=pretrain_checkpoint,
        encoder=encoder,
        predictor=None,
        context_encoder_key=context_encoder_key,
        target_encoder_key=target_encoder_key,
        target_encoder=None,
        load_predictor=False,
        load_encoder=load_encoder,
        replace_kw=["backbone.", "module."],
    )

    logger.info("Loading dreamer checkpoint...")
    dreamer_predictor = load_dreamer_predictor(
        dreamer_predictor_path=dreamer_checkpoint,
        dreamer_predictor=dreamer_predictor,
        replace_kw=["backbone.", "module."],
    )

    logger.info("Loading diffusion_head checkpoint...")
    diffusion_head = load_diffusion_head(
        diffusion_head_path=diffusion_head_checkpoint,
        diffusion_head=diffusion_head,
        replace_kw=("backbone.", "module."),
    )

    encoder = freeze_eval(encoder, dtype, device)
    dreamer_predictor = freeze_eval(dreamer_predictor, dtype, device)
    diffusion_head = freeze_eval(diffusion_head, dtype, device)

    world_model = WorldModel(
        encoder=encoder,
        dreamer_predictor=dreamer_predictor,
        diffusion_head=diffusion_head,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        tubelet_size=tubelet_size,
        normalize_reps=normalize_reps,
        device=device,
        dtype=dtype,
    )

    return world_model, dtype, mixed_precision


# ============================================================
# Client loop
# ============================================================

def connect_server(params: dict) -> socket.socket:
    deploy_cfg = params.get("deploy", {})
    server_ip = deploy_cfg.get("server_ip", "127.0.0.1")
    server_port = int(deploy_cfg.get("server_port", 9001))

    logger.info(f"Connecting to server: {server_ip}:{server_port}")
    return init_socket(server_ip, server_port)


def get_observation(sock: Optional[socket.socket], debugmode: bool) -> dict:
    if debugmode:
        return make_fake_data()

    msg = recv_msg(sock)

    if msg is None:
        raise ConnectionError("Socket closed by peer.")

    return pickle.loads(msg)


def format_action(action) -> list:
    """
    Convert diffusion output to server action.

    Supported shapes:
        [B, horizon, trajectory_dim]
        [horizon, trajectory_dim]
        [trajectory_dim]

    Server expects:
        [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    if torch.is_tensor(action):
        action = action.detach().float().cpu().numpy()

    action = np.asarray(action, dtype=np.float32)

    if action.ndim == 3:
        action = action[0, 0]
    elif action.ndim == 2:
        action = action[0]
    elif action.ndim == 1:
        pass
    else:
        raise ValueError(f"Unexpected action shape: {action.shape}")

    if action.shape[0] != 7:
        raise ValueError(f"Expected action dim=7, got shape={action.shape}")

    return action.tolist()


def deploy_loop(params: dict, debugmode: bool = False) -> None:
    world_model, dtype, mixed_precision = build_world_model(params)

    deploy_cfg = params.get("deploy", {})
    l1_threshold = float(deploy_cfg.get("l1_threshold", 1.0))
    queue_horizon = int(deploy_cfg.get("queue_horizon", 4))
    max_steps = int(deploy_cfg.get("max_steps", -1))

    sock = None if debugmode else connect_server(params)

    ref_pool = build_reference_loader(params)

    prev_goal = None
    obs_buffer = deque(maxlen=queue_horizon)

    step = 0

    try:
        while True:
            data = get_observation(sock=sock, debugmode=debugmode)

            if step > 0 and server_episode_finished(data):
                break

            current_img, pose = process_observation(data)

            current_ref, target_ref = select_reference_pair(
                world_model=world_model,
                current_img=current_img,
                prev_goal=prev_goal,
                obs_buffer=obs_buffer,
                ref_pool=ref_pool,
                dtype=dtype,
                mixed_precision=mixed_precision,
                l1_threshold=l1_threshold,
            )

            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                action, prev_goal = world_model(
                    current_img,
                    pose,
                    current_ref,
                    target_ref,
                )
                print(f"[CLIENT] prev_goal.shape: {prev_goal.shape}")

            action = format_action(action)

            if not debugmode:
                send_msg(sock, pickle.dumps(action))

            print("********************************")
            print(f"[CLIENT] step={step}")
            print(f"[CLIENT] action={action}")

            step += 1

            if max_steps > 0 and step >= max_steps:
                print(f"[DONE] reached max_steps={max_steps}")
                break

    finally:
        if sock is not None:
            sock.close()


# ============================================================
# Args
# ============================================================

def parse_bool(x):
    if isinstance(x, bool):
        return x

    x = str(x).lower()

    if x in ["true", "1", "yes", "y"]:
        return True

    if x in ["false", "0", "no", "n"]:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fname",
        type=str,
        default="configs.yaml",
    )

    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cuda:0"],
    )

    parser.add_argument(
        "--debugmode",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Run without socket server.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.fname, "r", encoding="utf-8") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    logger.info("loaded params...")
    deploy_loop(params, debugmode=args.debugmode)


if __name__ == "__main__":
    main()
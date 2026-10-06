#!/usr/bin/env python3
"""Does the AC world model know what an action does?  (stage 0's go/no-go, offline)

The test you would run by hand -- execute actions in the simulator, give the same
actions to the model, compare -- is what play.py episodes already are: random
actions EXECUTED in RLBench, with the frames they produced. So no simulator here:

For clips from those episodes, the model sees ONE real frame (deploy's regime)
and must say which action produced the next real frame:

  ranking   true action vs N-1 decoys (CEM-style uniform +/- maxnorm samples).
            Score each by L1(prediction, REAL next latent). top-1 / mean rank.
            Chance: top-1 = 1/N, rank pct = 50%. This is literally CEM's job.
  horizon   the same ranking over an H-step open-loop rollout (H actions,
            model feeding on its own predictions, as CEM rolls out).
  error     L1 to the real next latent with true / zero / flipped actions.

  python server/action_test.py --ckpt vjepa2_ac_repacked.pt --data data/play    # Meta, baseline
  python server/action_test.py --ckpt exp/stage0/latest.pt  --data data/play    # after stage 0

Use HELD-OUT data for the after-stage-0 number (e.g. --data data/play_val, play
episodes collected with a seed_master stage 0 never saw), or it measures memory.

Needs only the torch env and ~8 GB of GPU (encoder + predictor, bf16, no grad).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from app.vjepa_2_1_ac.dataset import ActionConditionedDataset  # noqa: E402
from app.vjepa_2_1_ac.transforms import make_transforms  # noqa: E402
from app.vjepa_2_1_ac.utils import (  # noqa: E402
    init_video_model, load_pretrained, vjepa_droid_encoder_args_from_cfg)
from src.utils.single_gpu import NoDDP  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(REPO, "vjepa2_ac_repacked.pt"))
    p.add_argument("--data", default=os.path.join(REPO, "data", "play"))
    p.add_argument("--cfg", default="configs/train/vjepa_2_1_ac.yaml",
                   help="model + data shape (written by prepare_configs.py)")
    p.add_argument("--data-fps", type=int, default=20, help="capture rate (1/dt)")
    p.add_argument("--clips", type=int, default=64)
    p.add_argument("--decoys", type=int, default=31)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--maxnorm", type=float, default=0.05, help="= deploy mpc.maxnorm")
    p.add_argument("--camera", default="right_shoulder_rgb")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vary", choices=("all", "xyz", "rot"), default="all",
                   help="what the decoys change. all: position + rotation. xyz: ONLY "
                        "position (rotation copied from the true action) -- does the "
                        "model know where the arm goes? rot: only rotation.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = yaml.safe_load(open(os.path.join(REPO, args.cfg)))
    m, d, aug = cfg["model"], cfg["data"], cfg["data_aug"]
    crop, patch, tubelet = int(d["crop_size"]), int(d["patch_size"]), int(d["tubelet_size"])
    tpf = (crop // patch) ** 2
    dev = torch.device("cuda:0")

    encoder, predictor = init_video_model(
        device=dev, patch_size=patch, max_num_frames=512, tubelet_size=tubelet,
        model_name=m["model_name"], crop_size=crop, pred_depth=m["pred_depth"],
        pred_num_heads=m.get("pred_num_heads"), pred_embed_dim=m["pred_embed_dim"],
        action_embed_dim=7, pred_is_frame_causal=m.get("pred_is_frame_causal", True),
        use_extrinsics=False, use_sdpa=True, use_silu=m.get("use_silu", False),
        use_pred_silu=m.get("use_pred_silu", False), wide_silu=m.get("wide_silu", True),
        use_rope=m.get("use_rope", False), uniform_power=m.get("uniform_power", False),
        use_activation_checkpointing=False, **vjepa_droid_encoder_args_from_cfg(m))
    encoder, predictor = NoDDP(encoder), NoDDP(predictor)     # `module.` keys, as saved
    load_pretrained(r_path=args.ckpt, encoder=encoder, predictor=predictor,
                    context_encoder_key="target_encoder", target_encoder=None,
                    load_predictor=True)
    encoder.eval()
    predictor.eval()

    ds = ActionConditionedDataset(
        dataset=args.data, camera_views=[args.camera], frameskip=1,
        frames_per_clip=args.horizon + 1, fps=int(d["fps"]), data_fps=args.data_fps,
        transform=make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=tuple(aug["random_resize_aspect_ratio"]),
            random_resize_scale=tuple(aug["random_resize_scale"]),
            reprob=0.0, auto_augment=False, motion_shift=False, crop_size=crop))
    if len(ds) == 0:
        sys.exit(f"no episodes under {args.data}")
    print(f"ckpt {args.ckpt}\ndata {args.data}: {len(ds)} episodes, "
          f"stride {-(-args.data_fps // int(d['fps']))} raw frames/action, "
          f"{args.decoys + 1} candidates, horizon {args.horizon}, decoys vary: {args.vary}\n")

    def encode(imgs):                       # [C,T,H,W] -> [T, tpf, D], layer-normed
        c = imgs.permute(1, 0, 2, 3).unsqueeze(2).repeat(1, 1, tubelet, 1, 1)
        h = encoder(c, masks=None, training=False)
        return F.layer_norm(h, (h.size(-1),))

    def rollout(z0, s0, acts):
        """z0 [tpf,D], s0 [7], acts [N,H,7] -> [N,H,tpf,D]. States integrate the
        actions (xyz add, small-angle rpy add, gripper absolute), as CEM does."""
        N, H, _ = acts.shape
        s = [s0.expand(N, 7)]
        for k in range(H - 1):
            nxt = s[-1].clone()
            nxt[:, :6] = nxt[:, :6] + acts[:, k, :6]
            nxt[:, 6] = acts[:, k, 6]
            s.append(nxt)
        states = torch.stack(s, 1)
        z = z0.unsqueeze(0).expand(N, -1, -1)
        outs = []
        for k in range(H):
            nxt = predictor(z, acts[:, :k + 1], states[:, :k + 1], None)[:, -tpf:]
            nxt = F.layer_norm(nxt, (nxt.size(-1),))
            outs.append(nxt)
            z = torch.cat([z, nxt], 1)
        return torch.stack(outs, 1)

    top1 = {1: [], args.horizon: []}
    rankpct = {1: [], args.horizon: []}
    err = {"true": [], "zero": [], "flip": []}
    for i in range(args.clips):
        imgs, acts, states = ds[np.random.randint(len(ds))]
        acts = torch.as_tensor(acts, dtype=torch.float, device=dev)        # [H,7]
        states = torch.as_tensor(states, dtype=torch.float, device=dev)    # [H+1,7]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = encode(imgs.to(dev))                                        # [H+1,tpf,D]
            dec = (torch.rand(args.decoys, acts.size(0), 7, device=dev) * 2 - 1) * args.maxnorm
            dec[..., 6] = acts[:, 6]                 # same gripper: rank the MOTION
            if args.vary == "xyz":                   # rotation can't give the answer away
                dec[..., 3:6] = acts[:, 3:6]
            elif args.vary == "rot":                 # position can't give it away
                dec[..., :3] = acts[:, :3]
            cand = torch.cat([acts.unsqueeze(0), dec], 0)                   # true = index 0
            pred = rollout(h[0], states[0], cand)                           # [N,H,tpf,D]
            for H in top1:
                dist = (pred[:, :H] - h[1:H + 1].unsqueeze(0)).abs().mean(dim=(1, 2, 3)).float()
                rank = int((dist < dist[0]).sum())
                top1[H].append(rank == 0)
                rankpct[H].append(rank / args.decoys)
            flip = acts.clone()
            flip[:, :6] = -flip[:, :6]
            zero = acts.clone()
            zero[:, :6] = 0
            for name, a in (("true", acts), ("zero", zero), ("flip", flip)):
                o = rollout(h[0], states[0], a[:1].unsqueeze(0))[0, 0]
                err[name].append(float((o - h[1]).abs().mean()))
        if i % 16 == 15:
            print(f"  {i + 1} clips: 1-step top-1 {np.mean(top1[1]):.2f}")

    N = args.decoys + 1
    print("\n" + "=" * 64)
    print(f"which of {N} actions produced the next real frame?  (chance top-1 "
          f"{1 / N:.2f}, rank 50%)")
    for H in top1:
        print(f"  {H}-step rollout:  top-1 {np.mean(top1[H]):.2f}   mean rank "
              f"{100 * np.mean(rankpct[H]):.0f}% of decoys beat it")
    print(f"\n1-step L1 to the real next latent (1 frame of context):")
    for k in err:
        print(f"  {k:5s} actions  {np.mean(err[k]):.4f}")
    print("=" * 64)
    t = np.mean(top1[1])
    if t >= 0.5:
        print("USES ACTIONS: the true action usually wins -- CEM has a signal.")
    elif t > 3 / N:
        print("WEAK: better than chance, not reliable. Stage 0 should raise this.")
    else:
        print("IGNORES ACTIONS: ~chance. CEM would return noise / a constant corner.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""What does stage 2 actually score on held-out data, IN ITS TRAINING REGIME?

server/dummy_test.py measured cost(GT action) ~= 0.36 through the deploy path,
against a stage 2 training loss of 0.102 -- a 3.5x gap on the same quantity
(L1 between predicted and true next latent, layer-normed). Two explanations,
needing opposite fixes:

  1. the model overfit / never learned. 402 pairs, 305M-param predictor.
     Stage 2 has to be retrained.
  2. deploy evaluates it outside its training regime, so the number is not
     comparable and the model may be fine. A config-level fix.

This settles it by running the EXACT training forward -- same dataloader class,
same forward_target, same teacher-forced predictor step, same loss_fn, copied
from app/vjepa_2_1_dreamer_ac/train.py -- against data/val instead of
data/train. Shapes are taken from the dataset rather than derived by hand,
because hand-deriving them did not reconcile: the predictor interleaves one
action token per latent frame (ac_predictor.py: torch.cat([a, s, x], dim=2)),
which requires T_actions == T_latent_frames, and the committed config appears to
give 8 latent frames against 3 actions. If that mismatch is real this script
raises it instead of hiding it.

It also asks the question dummy_test could only ask through deploy: does the
model USE the action, in the regime it was trained in? Same batch, three action
inputs:

  GT       - the real actions
  zero     - no motion
  shuffled - real actions, wrong order (same marginal distribution, no
             correspondence to the frames)

  loss(GT) << loss(zero) and << loss(shuffled)  -> conditioning works
  all three roughly equal                       -> the model ignores the action
  loss(GT) worst                                -> conditioning is anti-correlated,
                                                   which is what deploy showed

Runs in the torch env, no simulator:

  cd ~/Demo-JEPA && WANDB_MODE=disabled python server/val_loss.py
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

from app.vjepa_2_1_dreamer_ac.dataset import UnifiedPairedH5Dataset  # noqa: E402
from app.vjepa_2_1_dreamer_ac.deploy import build_world_model  # noqa: E402
from app.vjepa_2_1_dreamer_ac.transforms import make_transforms  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", default="configs/inference/deploy_vjepa_2_1.yaml",
                   help="supplies the model + checkpoints")
    p.add_argument("--train-cfg", default="configs/train/vjepa_2_1_dreamer_ac.yaml",
                   help="supplies the DATA shape actually used for training")
    p.add_argument("--data", default=os.path.join(REPO, "data", "val"))
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--camera", default="right_shoulder_rgb")
    # The shapes only balance when this is 1: images I=fpc, states after
    # [::frameskip] give S=ceil(I/frameskip), actions=S-1, latent frames=I, and
    # the predictor needs I-1 actions. init_data passes frameskip=tubelet_size
    # (=2), which cannot work. Exposed so the real value can be found by test.
    p.add_argument("--train-transform", action="store_true",
                   help="use training's image geometry (scale 1.777) instead of deploy's 1.0")
    p.add_argument("--frameskip", type=int, default=None,
                   help="dataset frameskip; default = tubelet_size, as init_data passes")
    args = p.parse_args()

    params = yaml.safe_load(open(os.path.join(REPO, args.fname)))
    tcfg = yaml.safe_load(open(os.path.join(REPO, args.train_cfg)))
    d = tcfg["data"]
    crop = int(d["crop_size"])
    tubelet = int(d["tubelet_size"])
    fpc = int(max(d["dataset_fpcs"]))
    fps, data_fps = int(d["fps"]), int(d["data_fps"])
    patch = int(d["patch_size"])
    normalize_reps = bool(tcfg["loss"].get("normalize_reps", True))
    loss_exp = float(tcfg["loss"].get("loss_exp", 1.0))
    tokens_per_frame = int((crop // patch) ** 2)

    print(f"train-regime shapes: fpc={fpc} tubelet={tubelet} fps={fps}/{data_fps} "
          f"crop={crop} tokens_per_frame={tokens_per_frame}")
    print(f"loss: normalize_reps={normalize_reps} exp={loss_exp}")
    print(f"data: {args.data}\n")

    # Training scaled every image by 1.777 (both ends of random_resize_scale are
    # equal, so it is deterministic, not an augmentation). deploy.py's
    # build_world_model uses scale (1.0, 1.0) -- a real train/deploy shift that
    # the encoder never saw. --train-transform reproduces training's geometry so
    # the loss is comparable to the reported 0.102.
    aug = tcfg.get("data_aug", {})
    if args.train_transform:
        ar = tuple(aug.get("random_resize_aspect_ratio", [1.0, 1.0]))
        sc = tuple(aug.get("random_resize_scale", [1.0, 1.0]))
        print(f"transform: TRAINING geometry  scale={sc} aspect={ar}")
    else:
        ar, sc = (1.0, 1.0), (1.0, 1.0)
        print(f"transform: DEPLOY geometry    scale={sc} aspect={ar}")
    transform = make_transforms(
        random_horizontal_flip=bool(aug.get("horizontal_flip", False)) if args.train_transform else False,
        random_resize_aspect_ratio=ar, random_resize_scale=sc,
        reprob=0.0, auto_augment=False, motion_shift=False, crop_size=crop,
    )

    ds = UnifiedPairedH5Dataset(
        dataset=args.data, camera_views=[args.camera],
        frameskip=(args.frameskip if args.frameskip else tubelet),
        frames_per_clip=fpc, fps=fps, data_fps=data_fps, transform=transform,
        camera_frame=False, primary_subdir="franka", reference_subdir="sawyer",
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, drop_last=True, num_workers=0)
    print(f"{len(ds)} held-out pairs\n")

    world_model, dtype, mixed = build_world_model(params)
    encoder, predictor = world_model.encoder, world_model.predictor
    device = world_model.device

    def forward_target(c, bsz):
        """Verbatim from train.py forward_target."""
        with torch.no_grad():
            c = (c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2)
                 .repeat(1, 1, tubelet, 1, 1))
            h = encoder(c, masks=None, training=False)
            h = h.view(bsz, -1, tokens_per_frame, h.size(-1)).flatten(1, 2)
            if normalize_reps:
                h = F.layer_norm(h, (h.size(-1),))
            return h

    def step_predictor(_z, _a, _s):
        _z = predictor(_z, _a, _s, None)
        if normalize_reps:
            _z = F.layer_norm(_z, (_z.size(-1),))
        return _z

    def loss_fn(z, h):
        """Verbatim from train.py loss_fn."""
        if z.shape[1] != h.shape[1]:
            _h = h[:, tokens_per_frame: z.size(1) + tokens_per_frame]
        else:
            _h = h
        return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp

    got = {"gt": [], "zero": [], "shuf": []}
    shapes_printed = False

    for bi, sample in enumerate(loader):
        if bi >= args.batches:
            break
        imgs, _ref, actions, states = sample
        bsz = imgs.size(0)
        imgs = imgs.to(device, non_blocking=True)
        actions = actions.to(device, dtype=torch.float, non_blocking=True)
        states = states.to(device, dtype=torch.float, non_blocking=True)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype, enabled=mixed):
            h = forward_target(imgs, bsz)
            n_latent = h.size(1) // tokens_per_frame
            if not shapes_printed:
                print(f"images {tuple(imgs.shape)}  ->  {n_latent} latent frames")
                print(f"actions {tuple(actions.shape)}  states {tuple(states.shape)}")
                if actions.size(1) != n_latent - 1:
                    print(f"  NOTE: {actions.size(1)} actions for {n_latent} latent "
                          f"frames; the predictor needs one per frame it predicts.")
                print()
                shapes_printed = True

            _z, _s = h[:, :-tokens_per_frame], states[:, :-1]
            variants = {
                "gt": actions,
                "zero": torch.zeros_like(actions),
                "shuf": actions[:, torch.randperm(actions.size(1), device=device)],
            }
            for name, a in variants.items():
                z = step_predictor(_z, a, _s)
                got[name].append(float(loss_fn(z, h).item()))

        print(f"[batch {bi}]  GT {got['gt'][-1]:.5f}   zero {got['zero'][-1]:.5f}   "
              f"shuffled {got['shuf'][-1]:.5f}")

    if not got["gt"]:
        sys.exit("ERROR: no batches ran")

    gt, zero, shuf = (float(np.mean(got[k])) for k in ("gt", "zero", "shuf"))
    print(f"\n{'=' * 62}")
    print(f"held-out loss, TRAINING regime")
    print(f"  GT actions        {gt:.5f}")
    print(f"  zero actions      {zero:.5f}   ({zero - gt:+.5f} vs GT)")
    print(f"  shuffled actions  {shuf:.5f}   ({shuf - gt:+.5f} vs GT)")
    print(f"  stage 2 reported  0.10200  (epoch 36, on data/train, WITH augmentation)")
    print(f"{'=' * 62}")

    if gt > 0.25:
        print("Held-out loss is far above the reported training loss.")
        print("=> the deploy-path number was NOT an artifact. The model does not")
        print("   generalise; stage 2 needs retraining. (402 pairs, 305M-param")
        print("   predictor -- overfitting is the obvious candidate.)")
    else:
        print("Held-out loss is close to the reported training loss.")
        print("=> the model is FINE in its training regime, and the deploy path is")
        print("   what breaks it. Fixable without retraining -- find the divergence")
        print("   between WorldModel.encode/__call__ and train.py's forward.")

    if gt <= zero and gt <= shuf:
        print("Action conditioning: GT scores best -- the model USES the action.")
    elif abs(zero - gt) < 0.01 and abs(shuf - gt) < 0.01:
        print("Action conditioning: all three near-identical -- the model IGNORES")
        print("the action. CEM has nothing to optimise, hence the constant corner.")
    else:
        print("Action conditioning: GT does NOT score best -- conditioning is")
        print("anti-correlated with reality, matching what deploy showed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

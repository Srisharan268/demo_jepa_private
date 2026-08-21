#!/usr/bin/env python3
"""SMOKE-TEST ONLY -- shrink stages 1 and 2 to fit a ~16GB card.

These are the reference Colab notebook's patches (cells 9 and 17), which ran on
a 15GB T4. They are memory hacks, NOT part of the real run:

  stage 1  frozen encoder -> bf16            (-4.8GB: params + DDP buckets)
           TRAINABLE dreamer_predictor -> bf16   *** REAL DEVIATION ***
                                             (-3.4GB: params, grads and both
                                              Adam moments all halve)
           bypass GradScaler                 (forced by the bf16 dreamer:
                                              _amp_foreach_non_finite_check_and
                                              _unscale_ has no BFloat16 kernel)
  stage 2  frozen encoder -> bf16
           frozen dreamer_predictor -> bf16
           TRAINABLE AC predictor -> bf16    *** REAL DEVIATION ***
           max_num_frames 512 -> 64          (17.4GB host-RAM causal mask)
           bypass GradScaler                 (needed only because of the bf16
                                              trainable predictor above)

The bf16 trainable modules (stage 1's dreamer_predictor, stage 2's AC
predictor) mean pure-bf16 training with no fp32 master weights -- small updates
round away. Acceptable for a plumbing test on a short run; never for a run you
report. Loss curves from a patched run are NOT comparable to a real one.

NEVER MERGE THIS INTO THE REAL BRANCH. Work on a throwaway branch:

    git checkout -b test-16gb
    python server/patch_16gb.py
    ...
    git checkout server-4gpu     # patches vanish

Idempotent. Usage:  python server/patch_16gb.py [--check]
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1 = os.path.join(REPO, "app", "vjepa_2_1_dreamer_predictor", "train.py")
S2 = os.path.join(REPO, "app", "vjepa_2_1_dreamer_ac", "train.py")
DEPLOY = os.path.join(REPO, "app", "vjepa_2_1_dreamer_ac", "deploy.py")

SCALER_BLOCK = "\n".join([
    "                if mixed_precision:",
    "                    scaler.scale(loss).backward()",
    "                    scaler.unscale_(optimizer)",
    "                else:",
    "                    loss.backward()",
    "                if mixed_precision:",
    "                    scaler.step(optimizer)",
    "                    scaler.update()",
    "                else:",
    "                    optimizer.step()",
    "",
])


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


def need(s, anchor, label, path):
    if s.count(anchor) != 1:
        sys.exit(f"ERROR: {os.path.basename(path)}: anchor '{label}' found "
                 f"{s.count(anchor)} times, expected 1.\nRepo differs from expectations -- stop.")


S1_ENC_CAST = "    encoder = encoder.to(dtype=torch.bfloat16)\n"
S1_DRM_CAST = "    dreamer_predictor = dreamer_predictor.to(dtype=torch.bfloat16)\n"

# GradScaler must be bypassed once dreamer_predictor is bf16: PyTorch has no
# BFloat16 kernel for _amp_foreach_non_finite_check_and_unscale_, so
# scaler.unscale_() raises NotImplementedError. Same reason patch_stage2
# bypasses it. bf16 needs no loss scaling anyway -- it shares fp32's exponent
# range, so the underflow that fp16 scaling exists to prevent cannot occur.
#
# Three separate sites in stage 1's accumulation loop, anchored so they match
# whether or not the micro_losses line has been switched to .detach().item().
S1_SCALER_SITES = [
    (
        "                    if mixed_precision:\n"
        "                        scaler.scale(loss).backward()\n"
        "                    else:\n"
        "                        loss.backward()\n",
        "                    loss.backward()\n",
        "scaler.scale/backward",
    ),
    (
        "                # Unscale once, on the fully-accumulated gradient\n"
        "                if mixed_precision:\n"
        "                    scaler.unscale_(optimizer)\n",
        "                # GradScaler bypassed (bf16): nothing to unscale.\n",
        "scaler.unscale_",
    ),
    (
        "                if mixed_precision:\n"
        "                    scaler.step(optimizer)\n"
        "                    scaler.update()\n"
        "                else:\n"
        "                    optimizer.step()\n",
        "                optimizer.step()\n",
        "scaler.step/update",
    ),
]


def patch_stage1(check):
    """Cast the frozen encoder AND the trainable dreamer_predictor to bf16.

    The dreamer cast is a REAL DEVIATION, the stage-1 twin of what
    patch_stage2 already does to the AC predictor: pure-bf16 training with no
    fp32 master weights, so small updates round away.

    It is what makes stage 1 fit 16GB. Adam allocates its state with
    torch.zeros_like(param), so a bf16 param halves params, grads, exp_avg and
    exp_avg_sq alike -- 6.8GB -> 3.4GB for a 424.7M-param model.

    Both casts must land before init_opt() and before the DDP wrap; the
    unfreeze_vit anchor (train.py ~line 214) precedes both, and
    dreamer_predictor already exists by then (init_video_model, ~line 194).

    The dreamer cast FORCES the GradScaler bypass -- see S1_SCALER_SITES.
    """
    s = read(S1)
    have_enc = S1_ENC_CAST.strip() in s
    have_drm = S1_DRM_CAST.strip() in s
    have_scaler_bypass = all(old not in s for old, _, _ in S1_SCALER_SITES)
    if have_enc and have_drm and have_scaler_bypass:
        return "stage 1: already patched"
    if check:
        missing = []
        if not have_enc:
            missing.append("encoder fp32")
        if not have_drm:
            missing.append("dreamer_predictor fp32")
        if not have_scaler_bypass:
            missing.append("GradScaler active")
        return f"stage 1: NOT patched ({', '.join(missing)})"

    if not (have_enc and have_drm):
        a = "    if unfreeze_vit:\n        target_encoder = deepcopy(encoder)"
        need(s, a, "unfreeze_vit/deepcopy", S1)
        # Insert only what is missing, so re-running after an older version of
        # this script tops up rather than duplicating the encoder cast.
        add = ("" if have_enc else S1_ENC_CAST) + ("" if have_drm else S1_DRM_CAST)
        s = s.replace(a, add + a, 1)

    for old, new, label in S1_SCALER_SITES:
        if new in s and old not in s:
            continue          # this site already bypassed
        need(s, old, label, S1)
        s = s.replace(old, new, 1)

    write(S1, s)
    return ("stage 1: patched (frozen encoder -> bf16, "
            "TRAINABLE dreamer_predictor -> bf16  *** REAL DEVIATION ***, "
            "GradScaler bypassed)")


def patch_stage2(check):
    s = read(S2)
    if "predictor = predictor.to(dtype=torch.bfloat16)" in s:
        return "stage 2: already patched"
    if check:
        return "stage 2: NOT patched"

    a1 = "    target_encoder = copy.deepcopy(encoder)\n    dreamer_predictor = init_dreamer_predictor("
    need(s, a1, "deepcopy/init_dreamer_predictor", S2)
    s = s.replace(a1, "    encoder = encoder.to(dtype=torch.bfloat16)\n"
                      "    predictor = predictor.to(dtype=torch.bfloat16)\n" + a1, 1)

    a2 = "        dreamer_predictor_fusion_type=dreamer_predictor_fusion_type,\n    )\n"
    need(s, a2, "dreamer_predictor_fusion_type", S2)
    s = s.replace(a2, a2 + "    dreamer_predictor = dreamer_predictor.to(dtype=torch.bfloat16)\n", 1)

    old = "        max_num_frames=512,"
    need(s, old, "max_num_frames=512", S2)
    s = s.replace(old, "        max_num_frames=64,", 1)

    need(s, SCALER_BLOCK, "scaler block", S2)
    s = s.replace(SCALER_BLOCK, "                loss.backward()\n                optimizer.step()\n", 1)

    write(S2, s)
    return "stage 2: patched (3x bf16, mask 64, GradScaler bypassed)"


def patch_deploy(check):
    s = read(DEPLOY)
    if "max_num_frames=64," in s:
        return "deploy: already patched"
    if check:
        return "deploy: NOT patched"
    old = "        max_num_frames=512,"
    need(s, old, "max_num_frames=512", DEPLOY)
    write(DEPLOY, s.replace(old, "        max_num_frames=64,", 1))
    return "deploy: patched (mask 64)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="report status, change nothing")
    args = p.parse_args()

    for fn in (patch_stage1, patch_stage2, patch_deploy):
        print(" ", fn(args.check))

    if not args.check:
        s1, s2 = read(S1), read(S2)
        print(f"\nverify: stage1 bf16 casts={s1.count('to(dtype=torch.bfloat16)')} (want 2), "
              f"stage1 scaler bypassed={'scaler.unscale_' not in s1}, "
              f"stage2 bf16 casts={s2.count('to(dtype=torch.bfloat16)')} (want 3), "
              f"mask64={'max_num_frames=64,' in s2}, "
              f"scaler bypassed={'scaler.unscale_' not in s2}")
        print("\nSMOKE-TEST ONLY. Do not merge. `git checkout server-4gpu` to revert.")


if __name__ == "__main__":
    main()

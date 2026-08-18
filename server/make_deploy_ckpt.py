#!/usr/bin/env python3
"""Extract a deploy-ready checkpoint from Stage 2's output.

deploy.py reads context_encoder_key=target_encoder and ["predictor"] from
`pretrain_checkpoint` (see configs/inference/deploy_vjepa_2_1.yaml), while
Stage 2 writes a much larger dict that also carries the optimizer state.

Loads with mmap=True so the multi-GB file is never fully materialised.

Usage:
  python server/make_deploy_ckpt.py /path/to/stage2/latest.pt /path/to/deploy.pt
"""
import os
import sys

import torch


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]

    if not os.path.exists(src):
        sys.exit(f"ERROR: not found: {src}")

    ck = torch.load(src, map_location="cpu", mmap=True)

    for key in ("target_encoder", "predictor"):
        if key not in ck:
            sys.exit(f"ERROR: {src} has no '{key}'. Keys present: {sorted(ck.keys())}")

    # deploy.py reads both context_encoder_key and target_encoder_key as
    # 'target_encoder', and load_encoder=true also wants 'encoder' -- point
    # both at the EMA weights, which is what the deploy config expects.
    torch.save(
        {
            "epoch": ck.get("epoch", 0),
            "target_encoder": ck["target_encoder"],
            "encoder": ck["target_encoder"],
            "predictor": ck["predictor"],
        },
        dst,
    )

    print(f"wrote {dst}  ({os.path.getsize(dst) / 1e9:.2f} GB, from {os.path.getsize(src) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Repack Meta's released V-JEPA 2.1-AC checkpoint for use as Stage 0 weights.

Download first (URL comes from the repo's own src/hub/backbones.py:8,17 --
VJEPA_BASE_URL + "vit_ac_giant" -> "vjepa2-ac-vitg"):

    wget --show-progress https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt

Three things this does that a naive dict-filter does not:

1. The raw file has top-level keys `encoder` and `predictor` -- there is NO
   `target_encoder`. Stage 1 and 2 read `target_encoder_key: target_encoder`,
   so it is aliased to `encoder` here. Filtering for a key that does not exist
   would silently produce a partial checkpoint.

2. Keys are renamed to `module.<name>`, stripping any existing `module.` and
   `backbone.` prefixes. The training scripts wrap models in DDP, which expects
   the `module.` prefix. Without this the state dict loads "successfully" while
   binding nothing -- the failure is silent and you only notice as a loss that
   never moves.

3. The optimizer state is dropped, which is most of the 11GB.

Dtype is preserved by default. Pass --bf16 to halve the file at the cost of
precision in the initial weights (that is what the Colab run did, purely to fit
in ~12.7GB of RAM -- not needed on a real node).

Usage:
  python server/repack_stage0.py vjepa2-ac-vitg.pt ~/vjepa2_ac_repacked.pt
"""
import argparse
import gc
import os
import sys

import torch


def convert(sd, to_bf16):
    out = {}
    for k, v in sd.items():
        name = "module." + k.replace("module.", "").replace("backbone.", "")
        out[name] = v.to(torch.bfloat16) if to_bf16 else v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", help="downloaded vjepa2-ac-vitg.pt")
    p.add_argument("dst", help="output path, e.g. ~/vjepa2_ac_repacked.pt")
    p.add_argument("--bf16", action="store_true",
                   help="cast weights to bfloat16 (smaller file, slight precision loss)")
    args = p.parse_args()

    src, dst = os.path.expanduser(args.src), os.path.expanduser(args.dst)
    if not os.path.exists(src):
        sys.exit(f"ERROR: not found: {src}\n"
                 f"Download it with:\n"
                 f"  wget --show-progress https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt")

    raw = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    print("top-level keys:", list(raw.keys()))

    for key in ("encoder", "predictor"):
        if key not in raw:
            sys.exit(f"ERROR: '{key}' missing from {src}. This is not the expected "
                     f"V-JEPA 2.1-AC checkpoint.")

    enc = convert(raw["encoder"], args.bf16)
    prd = convert(raw["predictor"], args.bf16)

    sample = next(iter(enc))
    print(f"encoder tensors : {len(enc)}")
    print(f"predictor tensors: {len(prd)}")
    print(f"sample key      : {sample}")
    print(f"dtype           : {next(iter(enc.values())).dtype}")

    if not sample.startswith("module."):
        sys.exit("ERROR: key renaming failed -- keys must start with 'module.' to bind under DDP.")

    # target_encoder aliases encoder: the raw file has no EMA copy, and both
    # context_encoder_key and target_encoder_key resolve to 'target_encoder'.
    torch.save({"epoch": 0, "encoder": enc, "target_encoder": enc, "predictor": prd}, dst)

    del raw, enc, prd
    gc.collect()
    print(f"\nwrote {dst}  ({os.path.getsize(dst) / 1e9:.2f} GB, from {os.path.getsize(src) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()

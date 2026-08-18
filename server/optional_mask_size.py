#!/usr/bin/env python3
"""OPTIONAL -- apply only if the node runs short on HOST RAM at Stage 2 startup.

app/vjepa_2_1_dreamer_ac/train.py builds the AC predictor with
max_num_frames=512. src/models/utils/modules.py then does:

    mask = torch.zeros(N, N).bool()

with N = (512/2) * (2 + 16*16) = 66,048. torch.zeros allocates fp32 BEFORE the
.bool() cast, i.e. 66,048^2 * 4 B = 17.4 GB in one allocation, leaving a
4.36 GB bool tensor resident -- per rank, so ~70 GB transient across 4 ranks.

This is HOST RAM, not VRAM: ac_predictor.py:156 slices the mask before moving
it to the GPU, so only ~4 MB ever reaches the device. If your node has plenty
of RAM, you do not need this patch.

Zero deviation either way: the mask is sliced to the live sequence length
regardless, and 64 frames gives (64/2) * 258 = 8,256 tokens against the ~2,064
actually used -- 4x margin. The mask is a plain attribute, not a registered
buffer, so checkpoints are unaffected.

Idempotent. Usage:  python server/optional_mask_size.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETS = [
    "app/vjepa_2_1_dreamer_ac/train.py",
    "app/vjepa_2_1_dreamer_ac/deploy.py",
]

OLD = "        max_num_frames=512,"
NEW = "        max_num_frames=64,"

for rel in TARGETS:
    path = os.path.join(REPO, rel)
    with open(path) as f:
        src = f.read()

    if NEW in src:
        print(f"{rel}: already patched")
        continue

    n = src.count(OLD)
    if n != 1:
        sys.exit(f"ERROR: {rel}: expected exactly 1 occurrence of {OLD!r}, found {n}")

    with open(path, "w") as f:
        f.write(src.replace(OLD, NEW, 1))
    print(f"{rel}: patched 512 -> 64")

print("\nNote: app/vjepa_2_1_ac/ and app/vjepa_2_1_imitation/ also contain")
print("max_num_frames=512, but those stages are not part of this run.")

#!/usr/bin/env python3
"""Build a side-by-side rollout video: source demo | policy execution.

Left  = the one-shot reference demonstration (the source embodiment, e.g. sawyer)
Right = what the policy actually did (the target embodiment, e.g. panda)

That pairing is the point of the paper -- cross-embodiment imitation from a
single demo -- so the comparison is the figure worth showing.

The two sequences differ in length, so both are resampled onto a common
timeline rather than truncated.

Usage:
  python server/make_video.py --frames rollouts/ep0 \\
      --reference data/val/push_button/sawyer/variation0_0000.hdf5 \\
      --out rollouts/ep0.gif
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
from PIL import Image

try:
    import imageio.v2 as imageio
except ImportError:  # older imageio
    import imageio


def load_frames(d):
    files = sorted(glob.glob(os.path.join(d, "**", "*.png"), recursive=True))
    files += sorted(glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True))
    if not files:
        sys.exit(f"ERROR: no frames in {d}\n"
                 f"Check that server.py ran with --save_image_dir and see its log.")
    return [np.asarray(Image.open(f).convert("RGB")) for f in sorted(files)]


def load_reference(h5_path, camera):
    key = f"observations/images/{camera}"
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            avail = []
            f.visit(lambda n: avail.append(n) if "images" in n else None)
            sys.exit(f"ERROR: '{key}' not in {h5_path}\nAvailable: {avail[:20]}")
        return [np.asarray(x) for x in f[key]]


def label(img, text, size):
    """Caption strip so the two panels are not ambiguous in a report."""
    from PIL import ImageDraw
    im = Image.fromarray(img).convert("RGB").resize(size)
    canvas = Image.new("RGB", (size[0], size[1] + 22), (18, 18, 18))
    canvas.paste(im, (0, 22))
    ImageDraw.Draw(canvas).text((6, 5), text, fill=(235, 235, 235))
    return np.asarray(canvas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, help="episode frame dir, e.g. rollouts/ep0")
    p.add_argument("--reference", required=True, help="reference demo .hdf5 used as the prompt")
    p.add_argument("--out", default=None)
    p.add_argument("--camera", default="right_shoulder_rgb")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--mp4", action="store_true", help="write .mp4 instead of .gif")
    args = p.parse_args()

    exec_frames = load_frames(args.frames)
    ref_frames = load_reference(args.reference, args.camera)
    print(f"reference: {len(ref_frames)} frames | execution: {len(exec_frames)} frames")

    out = args.out or os.path.join(args.frames.rstrip("/\\") + (".mp4" if args.mp4 else ".gif"))
    sz = (args.size, args.size)
    n = max(len(ref_frames), len(exec_frames))

    def at(seq, i):
        # resample onto the common timeline instead of truncating the longer one
        return seq[min(int(i * len(seq) / n), len(seq) - 1)]

    combined = [
        np.concatenate(
            [label(at(ref_frames, i), "reference demo (source)", sz),
             label(at(exec_frames, i), "policy execution (target)", sz)],
            axis=1,
        )
        for i in range(n)
    ]

    if args.mp4:
        imageio.mimsave(out, combined, fps=args.fps)
    else:
        imageio.mimsave(out, combined, fps=args.fps, loop=0)
    print(f"wrote {out}  ({n} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()

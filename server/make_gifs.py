#!/usr/bin/env python3
"""Convert rollout episode frame directories into GIFs.

Scans the given folder for episode subdirectories (ep0, ep1, …) containing
PNG/JPG frames and writes one GIF per episode to server/GIFs/.

Optionally accepts a --reference HDF5 file to produce a side-by-side
  [reference demo | policy execution] GIF instead of the bare rollout.

Usage (bare rollout GIFs):
    python server/make_gifs.py --folder rollouts/

Usage (side-by-side with reference demo):
    python server/make_gifs.py --folder rollouts/ \
        --reference data/train/push_button/sawyer/variation0_b1_chunk_0_0000.hdf5
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIF_DIR = os.path.join(REPO, "server", "GIFs")


def load_frames_from_dir(d):
    files = sorted(
        glob.glob(os.path.join(d, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True)
    )
    if not files:
        return []
    return [np.asarray(Image.open(f).convert("RGB")) for f in files]


def load_reference_frames(h5_path, camera="right_shoulder_rgb"):
    try:
        import h5py
    except ImportError:
        sys.exit("ERROR: h5py not installed. Run: pip install h5py")
    key = f"observations/images/{camera}"
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            avail = []
            f.visit(lambda n: avail.append(n) if "images" in n else None)
            sys.exit(f"ERROR: '{key}' not in {h5_path}\nAvailable image keys: {avail[:20]}")
        return [np.asarray(x) for x in f[key]]


def captioned(img, text, size):
    im = Image.fromarray(img).convert("RGB").resize(size)
    canvas = Image.new("RGB", (size[0], size[1] + 22), (18, 18, 18))
    canvas.paste(im, (0, 22))
    ImageDraw.Draw(canvas).text((6, 5), text, fill=(235, 235, 235))
    return np.asarray(canvas)


def frames_to_gif(frames, out_path, fps, size):
    sz = (size, size)
    result = [np.asarray(Image.fromarray(f).convert("RGB").resize(sz)) for f in frames]
    imageio.mimsave(out_path, result, fps=fps, loop=0)


def sidebyside_to_gif(ref_frames, exec_frames, out_path, fps, size):
    sz = (size, size)
    n = max(len(ref_frames), len(exec_frames))

    def at(seq, i):
        return seq[min(int(i * len(seq) / n), len(seq) - 1)]

    combined = [
        np.concatenate(
            [captioned(at(ref_frames, i), "reference demo (source)", sz),
             captioned(at(exec_frames, i), "policy execution (target)", sz)],
            axis=1,
        )
        for i in range(n)
    ]
    imageio.mimsave(out_path, combined, fps=fps, loop=0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", required=True,
                   help="Rollout folder containing ep0/, ep1/, ... subdirs")
    p.add_argument("--reference", default=None,
                   help="Optional HDF5 demo file for side-by-side GIF")
    p.add_argument("--camera", default="right_shoulder_rgb",
                   help="Camera key inside the HDF5 (default: right_shoulder_rgb)")
    p.add_argument("--out_dir", default=GIF_DIR,
                   help=f"Output directory (default: server/GIFs/)")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        sys.exit(f"ERROR: folder not found: {folder}")

    os.makedirs(args.out_dir, exist_ok=True)

    ep_dirs = sorted(
        d for d in glob.glob(os.path.join(folder, "ep*"))
        if os.path.isdir(d)
    )
    if not ep_dirs:
        sys.exit(f"ERROR: no ep* subdirectories found in {folder}")

    ref_frames = None
    if args.reference:
        print(f"Loading reference demo: {args.reference}")
        ref_frames = load_reference_frames(args.reference, args.camera)
        print(f"  {len(ref_frames)} reference frames")

    made, skipped = [], []
    for ep_dir in ep_dirs:
        ep_name = os.path.basename(ep_dir)
        exec_frames = load_frames_from_dir(ep_dir)
        if not exec_frames:
            print(f"  [skip] {ep_name}: no frames found")
            skipped.append(ep_name)
            continue

        folder_tag = os.path.basename(folder.rstrip("/\\"))
        out_path = os.path.join(args.out_dir, f"{folder_tag}_{ep_name}.gif")

        if ref_frames is not None:
            sidebyside_to_gif(ref_frames, exec_frames, out_path, args.fps, args.size)
        else:
            frames_to_gif(exec_frames, out_path, args.fps, args.size)

        print(f"  wrote {out_path}  ({len(exec_frames)} frames)")
        made.append(out_path)

    print(f"\nDone - {len(made)} GIF(s) written to {args.out_dir}")
    if skipped:
        print(f"Skipped (no frames): {skipped}")


if __name__ == "__main__":
    main()

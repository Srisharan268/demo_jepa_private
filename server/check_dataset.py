#!/usr/bin/env python3
"""Audit a paired dataset -- and, on request, repair it without deleting anything.

Checks, all of which were found broken in the 402-pair push_button dataset:

  pairing      every franka episode has a sawyer episode of the same name
  duplicates   episodes generated from the same seed are bit-identical copies.
               The old collector's seed was linear in (master, pair), so chunks
               run with consecutive masters collided: 150 of 402 were copies.
  leakage      a held-out episode whose copy is also in train is not held out:
               30 of 40 val episodes were.
  labels       the old collector stored the COMMANDED source pose as qpos for
               every robot, so franka qpos == sawyer qpos in every pair and the
               action labels never described what the franka actually did.
  replay       (new collector only) every replay completed the task and tracked
               the demo within its limit.
  dt           capture rate, which prepare_configs.py uses to derive data_fps.

Repairs (both only MOVE files):

  --merge-val  move data/val back into data/train, so the pool can be
               de-duplicated and re-split cleanly
  --dedup      keep one episode pair per seed; move the other copies (both
               robots) to data/duplicates/

Clean procedure for the existing dataset:

  python server/check_dataset.py --merge-val --dedup
  python server/split_dataset.py --val 30
  python server/check_dataset.py            # expect: no duplicates, no leakage

Needs only h5py and numpy.
"""
import argparse
import collections
import glob
import os
import shutil
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOTS = ("franka", "sawyer")


def scan(root):
    """{(split, task, name): {robot: path}} for every episode under root."""
    eps = collections.defaultdict(dict)
    for split in ("train", "val"):
        for path in glob.glob(os.path.join(root, split, "*", "*", "*.hdf5")):
            task, robot = path.split(os.sep)[-3], path.split(os.sep)[-2]
            if robot in ROBOTS:
                name = os.path.splitext(os.path.basename(path))[0]
                eps[(split, task, name)][robot] = path
    return eps


def attrs(path):
    with h5py.File(path, "r") as f:
        return dict(f.attrs), np.asarray(f["observations/qpos"])


def move(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        sys.exit(f"ERROR: refusing to overwrite {dst}")
    shutil.move(src, dst)


def merge_val(root):
    moved = 0
    for path in glob.glob(os.path.join(root, "val", "*", "*", "*.hdf5")):
        task, robot, fname = path.split(os.sep)[-3:]
        move(path, os.path.join(root, "train", task, robot, fname))
        moved += 1
    print(f"merged {moved} val file(s) back into train/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.join(REPO, "data"))
    p.add_argument("--merge-val", action="store_true")
    p.add_argument("--dedup", action="store_true")
    args = p.parse_args()

    if args.merge_val:
        merge_val(args.root)

    eps = scan(args.root)
    if not eps:
        sys.exit(f"ERROR: no episodes under {args.root}/{{train,val}}/<task>/{{franka,sawyer}}/")

    problems = 0

    # -- pairing
    unpaired = [k for k, v in eps.items() if set(v) != set(ROBOTS)]
    print(f"episodes: {len(eps)}  "
          f"({collections.Counter(k[0] for k in eps)}) across tasks "
          f"{sorted({k[1] for k in eps})}")
    if unpaired:
        problems += 1
        print(f"[FAIL] {len(unpaired)} episode(s) missing a robot, e.g. {unpaired[:3]}")
    else:
        print("[ok]   every episode has both franka and sawyer")

    # -- per-episode attributes (read the franka side; both sides share seed/dt)
    seeds = collections.defaultdict(list)
    dts, sources, shared_qpos, n_new, bad_replay = set(), set(), 0, 0, []
    lengths = collections.defaultdict(list)       # task -> episode lengths
    track = collections.defaultdict(list)         # robot -> max tracking error (m)
    for key, robots in sorted(eps.items()):
        if "franka" not in robots:
            continue
        a, qf = attrs(robots["franka"])
        seeds[(key[1], int(a["episode_seed_used"]))].append(key)
        dts.add(float(a.get("dt", -1)))
        lengths[key[1]].append(len(qf))
        sources.add(str(a.get("source_robot")))
        if "sawyer" in robots:
            _, qs = attrs(robots["sawyer"])
            shared_qpos += int(qf.shape == qs.shape and np.array_equal(qf, qs))
        if "replay_success" in a:
            n_new += 1
            for r, path in robots.items():
                ra, _ = attrs(path)
                if not bool(ra["replay_success"]):
                    bad_replay.append((key[2], r))
                track[r].append(float(ra["replay_max_pos_err"]))

    print(f"[info] dt {sorted(dts)}  (1/dt = capture rate; prepare_configs derives data_fps from it)")
    print(f"[info] source robot(s): {sorted(sources)}  (recover_rng --robot-subdir must point at it)")

    # -- episode length. Stage 2 samples 8 frames spaced ceil(data_fps/fps) raw
    # frames apart and needs one more for the last action: at 20 Hz / 5 fps that
    # is 33. The loader silently skips shorter episodes -- and if EVERY episode
    # of a dataset is short it resamples forever.
    need = 8 * 4 + 1
    for task, ls in sorted(lengths.items()):
        short = sum(l < need for l in ls)
        tag = "[ok]  " if short == 0 else ("[FAIL]" if short == len(ls) else "[WARN]")
        if short:
            problems += tag == "[FAIL]"
        print(f"{tag} {task}: episode length min {min(ls)} / median {int(np.median(ls))} "
              f"frames; {short}/{len(ls)} shorter than the {need} stage 2 needs")

    # -- tracking (recorded, not gated: contact legitimately blocks the arm)
    for r, errs in sorted(track.items()):
        e = np.array(errs) * 1000
        print(f"[info] {r} replay max tracking error: median {np.median(e):.1f} mm, "
              f"p90 {np.percentile(e, 90):.1f} mm, worst {e.max():.1f} mm")

    # -- labels
    if shared_qpos:
        # A warning, not a failure: the data is usable, just imperfectly labelled.
        print(f"[WARN] {shared_qpos} pair(s) have franka qpos IDENTICAL to sawyer qpos: "
              f"old-collector data, whose action labels are the COMMANDED source "
              f"poses, not what each robot did. Usable, but re-collect for correct labels.")
    else:
        print("[ok]   each robot's qpos is its own (achieved-pose labels)")

    # -- replay quality (new collector only)
    if n_new:
        if bad_replay:
            problems += 1
            print(f"[FAIL] {len(bad_replay)} replay(s) did not complete the task, e.g. {bad_replay[:3]}")
        else:
            print(f"[ok]   all {n_new} new-collector episodes completed the task on every robot")

    # -- duplicates and leakage
    dup_groups = {s: v for s, v in seeds.items() if len(v) > 1}
    n_dup = sum(len(v) - 1 for v in dup_groups.values())
    leaks = [v for v in dup_groups.values() if len({k[0] for k in v}) > 1]
    if dup_groups:
        problems += 1
        print(f"[FAIL] {n_dup} duplicate episode(s): {len(seeds)} unique seeds among "
              f"{sum(len(v) for v in seeds.values())} episodes")
        ex = next(iter(dup_groups.values()))
        print(f"       e.g. {[(k[0], k[2]) for k in ex]}")
    else:
        print("[ok]   no duplicate seeds")
    if leaks:
        n_val = sum(1 for v in leaks for k in v if k[0] == "val")
        print(f"[FAIL] {n_val} val episode(s) have a copy in train -- NOT held out")
    elif dup_groups:
        print("[ok]   no duplicates span train and val")

    # -- repair
    if args.dedup and dup_groups:
        moved = 0
        for group in dup_groups.values():
            # Keep one copy, preferring train so the held-out set is rebuilt
            # fresh by split_dataset.py rather than inherited.
            keep = sorted(group, key=lambda k: (k[0] != "train", k[2]))[0]
            for key in group:
                if key == keep:
                    continue
                for robot, path in eps[key].items():
                    move(path, os.path.join(args.root, "duplicates", key[1], robot,
                                            os.path.basename(path)))
                moved += 1
        print(f"\nmoved {moved} duplicate episode pair(s) to "
              f"{os.path.join(args.root, 'duplicates')}/ (nothing deleted)")
        print("re-run this script to confirm, then split_dataset.py for a fresh val set.")
    elif dup_groups:
        print("\nrun with --merge-val --dedup to repair (moves files only).")

    return 1 if problems and not args.dedup else 0


if __name__ == "__main__":
    sys.exit(main())

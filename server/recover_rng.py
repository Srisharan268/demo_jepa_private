#!/usr/bin/env python3
"""Rebuild the deleted rng_state.pkl / meta.json from the seeds stored in the HDF5s.

Rollouts currently reset the simulator to a RANDOM scene while the reference
demo comes from a specific held-out episode, so the policy is servoing toward a
button placement that does not exist in the live scene. Fixing that needs
`server.py --episode_dir`, which needs `rng_state.pkl` in the episode's parent
chain. Collection did write those files (retarget.py: save_rng_state), but the
pair_root tree was deleted when the dataset was flattened into
data/<split>/<task>/<robot>/*.hdf5.

They are recoverable. save_demo_h5 stamps `episode_seed_used` and `variation`
into each HDF5, and the RNG path is deterministic:

    set_seed(seed) -> task_env.reset() -> get_demos(1) -> demo.random_seed

so re-running generate_source_episode() with the stored seed reproduces the same
rng_state. Only the demo is regenerated -- no images are written -- so this is
far cheaper than re-collecting.

*** THE RESULT IS VERIFIED, NOT TRUSTED. ***
`observations/qpos` in the stored file is built directly from the returned
actions (io_utils.build_qpos_qvel_action), so a regenerated episode must
reproduce it exactly. Anything that perturbs RNG consumption -- a different
number of retries inside generate_source_episode's attempt loop, different
arm velocity/acceleration limits, a different RLBench build -- shows up as a
qpos mismatch. Episodes that fail verification are REPORTED AND SKIPPED; no
rng_state.pkl is written for them, because a wrong scene is worse than a
random one (it looks correct and is not).

Runs in the rlbench env (pyrep + RLBench), needs a display. The interpreter and
CoppeliaSim paths differ per machine -- take them from PY_SIM and
COPPELIASIM_ROOT at the top of server/run_rollout.py on THIS branch rather than
copying them from anywhere else (the 4080 uses a relocated prebuilt sim under
/home/cobot/simenv, the cloud image used /opt):

  eval "$(python - <<'EOF'
import re
s = open("server/run_rollout.py").read()
for k in ("PY_SIM", "COPPELIASIM_ROOT"):
    print(f'{k}={re.search(rf"^{k} = \"([^\"]+)\"", s, re.M).group(1)}')
EOF
)"
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
  export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
  export DISPLAY=:99          # Xvfb, as run_rollout.py starts it
  $PY_SIM server/recover_rng.py --task push_button

Then point the rollout at a recovered scene:

  server.py --episode_dir data/scenes/push_button/variation0_0000 ...
"""
import argparse
import json
import os
import pickle
import sys

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# retarget.py and friends use bare top-level imports (`from action_utils import
# ...`), so their own directory has to be importable, not just the repo root.
sys.path.insert(0, os.path.join(REPO, "scripts", "rlbench_tools"))

from config import RetargetConfig  # noqa: E402
from io_utils import build_qpos_qvel_action  # noqa: E402
from retarget import generate_source_episode  # noqa: E402


def episode_attrs(path):
    with h5py.File(path, "r") as f:
        a = dict(f.attrs)
        qpos = np.asarray(f["observations/qpos"])
    missing = [k for k in ("episode_seed_used", "variation", "task") if k not in a]
    if missing:
        raise KeyError(f"{os.path.basename(path)}: missing attrs {missing}")
    return a, qpos


def compare(stored_qpos, actions):
    """Exact match required. Returns (ok, detail)."""
    T = len(stored_qpos)
    if len(actions) < T:
        return False, f"regenerated {len(actions)} steps, stored has {T}"

    regen, _, _ = build_qpos_qvel_action(np.asarray(actions, dtype=np.float32), T)
    regen = np.asarray(regen, dtype=np.float32)
    stored = np.asarray(stored_qpos, dtype=np.float32)

    if regen.shape != stored.shape:
        return False, f"shape {regen.shape} vs stored {stored.shape}"
    if np.array_equal(regen, stored):
        return True, "exact"

    diff = float(np.max(np.abs(regen - stored)))
    # Not a tolerance check -- a genuinely identical trajectory is bit-identical
    # here, both sides being float32 built by the same function. A small but
    # nonzero diff means the scene drifted, which is exactly the failure this
    # script exists to catch.
    return False, f"max abs diff {diff:.6g}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.path.join(REPO, "data", "val"))
    p.add_argument("--task", default=None, help="default: every task under --data-root")
    p.add_argument("--robot-subdir", default="franka",
                   help="side holding the SOURCE demo (attrs robot == source_robot)")
    p.add_argument("--out", default=os.path.join(REPO, "data", "scenes"))
    # Not stamped into the HDF5, so they cannot be read back. If verification
    # fails for every episode, these are the first thing to suspect.
    p.add_argument("--arm-max-velocity", type=float, default=1.0)
    p.add_argument("--arm-max-acceleration", type=float, default=4.0)
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--limit", type=int, default=0, help="stop after N episodes (0 = all)")
    args = p.parse_args()

    if not os.path.isdir(args.data_root):
        sys.exit(f"ERROR: no such directory: {args.data_root}")

    tasks = ([args.task] if args.task else
             sorted(d for d in os.listdir(args.data_root)
                    if os.path.isdir(os.path.join(args.data_root, d))))
    if not tasks:
        sys.exit(f"ERROR: no task directories under {args.data_root}")

    episodes = []
    for task in tasks:
        d = os.path.join(args.data_root, task, args.robot_subdir)
        if not os.path.isdir(d):
            sys.exit(f"ERROR: missing {d}")
        episodes += [os.path.join(d, f) for f in sorted(os.listdir(d))
                     if f.endswith((".hdf5", ".h5"))]
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        sys.exit("ERROR: no episodes found")

    print(f"recovering {len(episodes)} episode(s) -> {args.out}\n")

    ok_n = fail_n = 0
    failures = []

    for i, path in enumerate(episodes):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            a, stored_qpos = episode_attrs(path)
        except KeyError as e:
            print(f"[{i + 1}/{len(episodes)}] {name}: SKIP -- {e}", flush=True)
            fail_n += 1
            failures.append((name, str(e)))
            continue

        task = str(a["task"])
        seed = int(a["episode_seed_used"])
        variation = int(a["variation"])
        source_robot = str(a.get("source_robot", "panda"))

        if str(a.get("robot", source_robot)) != source_robot:
            print(f"[{i + 1}/{len(episodes)}] {name}: SKIP -- robot={a.get('robot')!r} is "
                  f"not the source ({source_robot!r}); point --robot-subdir at the "
                  f"source side", flush=True)
            fail_n += 1
            failures.append((name, "not the source robot"))
            continue

        cfg = RetargetConfig(
            task=task,
            source_robot=source_robot,
            renderer=str(a.get("renderer", "opengl")),
            dt=float(a.get("dt", 0.05)),
            static_positions=bool(a.get("static_positions", False)),
            headless=True,
            image_width=args.image_width,
            image_height=args.image_height,
            arm_max_velocity=args.arm_max_velocity,
            arm_max_acceleration=args.arm_max_acceleration,
        )

        print(f"[{i + 1}/{len(episodes)}] {name}  seed={seed} var={variation} ... ",
              end="", flush=True)
        try:
            actions, rng_state, actual_var = generate_source_episode(cfg, variation, seed)
        except Exception as e:
            print(f"FAILED to regenerate: {e}", flush=True)
            fail_n += 1
            failures.append((name, f"regeneration error: {e}"))
            continue

        good, detail = compare(stored_qpos, actions)
        if not good:
            print(f"MISMATCH ({detail}) -- not writing", flush=True)
            fail_n += 1
            failures.append((name, detail))
            continue

        pair_root = os.path.join(args.out, task, name)
        os.makedirs(pair_root, exist_ok=True)
        with open(os.path.join(pair_root, "rng_state.pkl"), "wb") as f:
            pickle.dump(rng_state, f)
        # server.py's load_meta only reads `task` and `variation`, and tolerates
        # the file being absent, so a minimal record is enough.
        with open(os.path.join(pair_root, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "task": task,
                "variation": int(actual_var),
                "episode_seed_used": seed,
                "source_episode": os.path.relpath(path, REPO),
                "recovered_by": "server/recover_rng.py",
                "verified": "qpos exact match",
            }, f, indent=2)

        print(f"OK ({detail}) -> {os.path.relpath(pair_root, REPO)}", flush=True)
        ok_n += 1

    print(f"\n{'=' * 62}")
    print(f"verified and written : {ok_n}/{len(episodes)}")
    print(f"failed / skipped     : {fail_n}/{len(episodes)}")
    if failures:
        print("\nfailures:")
        for name, why in failures:
            print(f"  {name:32s} {why}")
        print("\nIf EVERY episode mismatched, the regeneration parameters differ from\n"
              "collection -- try --arm-max-velocity / --arm-max-acceleration, or check\n"
              "that this is the same RLBench build the data was collected with.\n"
              "If only some failed, those episodes hit a different retry count and are\n"
              "simply unrecoverable this way; use the ones that verified.")
    if ok_n:
        print(f"\nUse a recovered scene:\n"
              f"  server.py --episode_dir {os.path.join(args.out, tasks[0], '<episode>')} ...")
    print(f"{'=' * 62}")
    return 0 if ok_n else 1


if __name__ == "__main__":
    sys.exit(main())

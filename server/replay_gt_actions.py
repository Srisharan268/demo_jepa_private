#!/usr/bin/env python3
"""Replay a demo's OWN actions through server.py, with no model involved.

This separates two hypotheses that rollout logs cannot distinguish:

  A. the action pipeline is sound and the POLICY is bad
  B. there is a convention/scaling bug between the dataset's actions and what
     server.py expects, in which case no amount of training or tuning helps

Ground-truth actions are reconstructed exactly as the training dataloader does
(dataset.py: quaternion_to_euler then poses_to_diffs) -- xyz deltas plus
euler-angle deltas from relative rotation matrices, plus absolute gripper. So a
success here means the pipeline can execute the very actions the model was
trained to imitate.

Run server.py yourself first, then this against it:

  cd scripts/rlbench_tools && $PY_SIM -u server.py --host 127.0.0.1 --port 9001 \\
      --task push_button --robot panda --image_size 256 256 \\
      --renderer opengl --headless --save_image_dir <abs dir> &

  python server/replay_gt_actions.py --episode data/val/push_button/franka/<f>.hdf5

Needs only numpy/scipy/h5py -- no torch, no GPU.
"""
import argparse
import glob
import os
import pickle
import socket
import struct
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def send_msg(sock, data):
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)


def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(sock):
    raw = recvall(sock, 4)
    if not raw:
        return None
    return recvall(sock, struct.unpack(">I", raw)[0])


def gt_actions(path):
    """Exactly dataset.py's quaternion_to_euler -> poses_to_diffs."""
    with h5py.File(path) as h:
        poses = np.array(h["observations/qpos"], dtype=np.float64)
    if poses.shape[-1] != 7:
        xyz, quat, grip = poses[:, :3], poses[:, 3:7], poses[:, -1:]
        euler = np.stack([Rotation.from_quat(q).as_euler("xyz") for q in quat])
        poses = np.concatenate([xyz, euler, grip], axis=1)
    xyz, thetas = poses[:, :3], poses[:, 3:6]
    mats = [Rotation.from_euler("xyz", t).as_matrix() for t in thetas]
    xyz_diff = xyz[1:] - xyz[:-1]
    ang_diff = np.stack([
        Rotation.from_matrix(mats[t + 1] @ mats[t].T).as_euler("xyz")
        for t in range(len(mats) - 1)
    ])
    return np.concatenate([xyz_diff, ang_diff, poses[:, -1:][1:]], axis=1), poses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", default=None, help="hdf5; default = first val episode")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9001)
    p.add_argument("--steps", type=int, default=40)
    a = p.parse_args()

    ep = a.episode or sorted(glob.glob("data/val/push_button/franka/*.hdf5"))[0]
    acts, poses = gt_actions(ep)
    print(f"episode {os.path.basename(ep)}: {len(acts)} actions")
    print(f"  demo start xyz {poses[0][:3].round(4)}  end {poses[-1][:3].round(4)}")
    print(f"  |delta| mean {np.abs(acts[:, :3]).mean(0).round(5)}  "
          f"max {np.abs(acts[:, :3]).max(0).round(5)}\n")

    s = socket.socket()
    s.connect((a.host, a.port))
    init = pickle.loads(recv_msg(s))
    print(f"connected; initial obs keys: {sorted(init)[:6]}...\n")

    n = min(a.steps, len(acts))
    fails = 0
    for i in range(n):
        send_msg(s, pickle.dumps(acts[i].astype(np.float32)))
        reply = recv_msg(s)
        if reply is None:
            print(f"step {i}: server closed the connection")
            break
        r = pickle.loads(reply)
        failed, success = r.get("failed"), r.get("success")
        fails += bool(failed)
        if i < 5 or failed or success:
            print(f"  step {i:3d}  act {acts[i][:3].round(5)}  "
                  f"failed={failed}  success={success}  reward={r.get('reward')}")
        if success:
            print(f"\nSUCCESS at step {i}")
            break
    s.close()

    print(f"\n{'=' * 60}")
    print(f"IK/exec failures: {fails}/{n}")
    if fails == 0:
        print("PIPELINE OK -- ground-truth actions execute cleanly.")
        print("=> the action path is sound; remaining rollout failures are POLICY.")
    elif fails >= n * 0.5:
        print("*** GROUND TRUTH ALSO FAILS ***")
        print("=> convention/scaling bug between the dataset and server.py.")
        print("   No amount of training or maxnorm tuning fixes this.")
    else:
        print("Partial failures on ground truth -- inspect the per-step output above.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

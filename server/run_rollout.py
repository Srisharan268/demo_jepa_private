#!/usr/bin/env python3
"""Closed-loop rollout evaluation for Stage 2 (Demo-JEPA).

Two processes, two conda envs, talking over localhost:

  server.py  -- pyrep + rlbench + CoppeliaSim, NO torch      (env: PY_SIM)
  deploy.py  -- torch + CUDA, runs the CEM/MPC policy        (this env)

They meet only on the socket, so the dependency stacks never interact. The
simulator renders offscreen under Xvfb, and writes RGB frames via server.py's
--save_image_dir, which is where the rollout video comes from -- deploy.py
itself saves nothing.

Per episode: start the simulator, run the policy, tear down, record success.

Usage:
  python server/run_rollout.py --episodes 10 --task push_button
"""
import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------
# EDIT THESE
# ----------------------------------------------------------------------------
PY_SIM = "/opt/conda/envs/rlbench/bin/python"   # python with pyrep+rlbench
COPPELIASIM_ROOT = "/opt/CoppeliaSim"
DISPLAY = ":99"
# ----------------------------------------------------------------------------

DEPLOY_CFG = "configs/inference/deploy_vjepa_2_1.yaml"
PORT = 9001


def sim_env():
    e = dict(os.environ)
    e["COPPELIASIM_ROOT"] = COPPELIASIM_ROOT
    e["LD_LIBRARY_PATH"] = f"{e.get('LD_LIBRARY_PATH', '')}:{COPPELIASIM_ROOT}"
    e["QT_QPA_PLATFORM_PLUGIN_PATH"] = COPPELIASIM_ROOT
    e["DISPLAY"] = DISPLAY
    e["PYTHONUNBUFFERED"] = "1"
    return e


def torch_env():
    e = dict(os.environ)
    e["PYTHONPATH"] = REPO + (":" + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    e["PYTHONUNBUFFERED"] = "1"
    e.setdefault("WANDB_MODE", "disabled")
    return e


def ensure_xvfb():
    """Headless rendering needs a virtual display; CoppeliaSim will not start without one."""
    if subprocess.run(f"xdpyinfo -display {DISPLAY}", shell=True,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return "already running"
    subprocess.Popen(["Xvfb", DISPLAY, "-screen", "0", "1400x900x24",
                      "-ac", "+extension", "GLX", "+render", "-noreset"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    return "started"


def wait_for_port(port, timeout=180):
    """Poll instead of sleeping a fixed interval -- CoppeliaSim startup varies a lot."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(2)
    return False


def kill_stale():
    subprocess.run("pkill -f coppeliaSim; pkill -f rlbench_tools/server.py",
                   shell=True, stderr=subprocess.DEVNULL)
    time.sleep(2)


def run_episode(ep, args):
    frames_dir = os.path.join(args.out, f"ep{ep}")
    os.makedirs(frames_dir, exist_ok=True)
    server_log = os.path.join(args.out, f"server_ep{ep}.log")

    kill_stale()
    ensure_xvfb()

    cmd = [
        PY_SIM, "-u", "server.py",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--task", args.task, "--robot", args.robot,
        "--image_size", "256", "256",
        "--renderer", "opengl", "--headless",
        "--save_image_dir", frames_dir,
    ]
    with open(server_log, "w") as log:
        srv = subprocess.Popen(cmd, cwd=os.path.join(REPO, "scripts", "rlbench_tools"),
                               env=sim_env(), stdout=log, stderr=subprocess.STDOUT)

    print(f"  simulator starting (CoppeliaSim, up to {args.timeout}s)...", flush=True)
    if not wait_for_port(PORT, timeout=args.timeout):
        srv.terminate()
        print(f"  ERROR: server never opened :{PORT}. See {server_log}", flush=True)
        return False, 0

    print("  running policy (CEM: 200 samples x 50 steps per env step -- slow)", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "app.vjepa_2_1_dreamer_ac.deploy", "--fname", DEPLOY_CFG],
        cwd=REPO, env=torch_env(), capture_output=True, text=True,
    )

    srv.terminate()
    try:
        srv.wait(timeout=30)
    except subprocess.TimeoutExpired:
        srv.kill()
    kill_stale()

    log_text = open(server_log, errors="replace").read()
    # server.py prints "success={bool}" each substep. Match the value, not the
    # bare word -- "success=False" contains "success".
    success = bool(re.search(r"success=True", log_text))

    n_frames = len([f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg"))])

    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-15:])
        print(f"  deploy.py exited {proc.returncode}:\n{tail}", flush=True)

    return success, n_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--robot", type=str, default="panda", help="target embodiment")
    p.add_argument("--out", type=str, default=os.path.join(REPO, "rollouts"))
    p.add_argument("--timeout", type=int, default=180, help="seconds to wait for CoppeliaSim")
    p.add_argument("--fresh", action="store_true", help="wipe --out first")
    args = p.parse_args()

    if args.fresh and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(PY_SIM):
        sys.exit(f"ERROR: PY_SIM not found: {PY_SIM}\nEdit the path at the top of this file.")

    print(f"xvfb: {ensure_xvfb()}   task: {args.task}   episodes: {args.episodes}")

    results = []
    for ep in range(args.episodes):
        print(f"\n=== episode {ep + 1}/{args.episodes} ===", flush=True)
        ok, n = run_episode(ep, args)
        results.append(ok)
        print(f"  -> {'SUCCESS' if ok else 'FAIL'}   ({n} frames)", flush=True)

    n_ok = sum(results)
    print(f"\n{'=' * 46}")
    print(f"SUCCESS RATE: {n_ok}/{len(results)} = {100 * n_ok / max(len(results), 1):.1f}%")
    print(f"frames under: {args.out}")
    print(f"{'=' * 46}")

    with open(os.path.join(args.out, "results.txt"), "w") as f:
        for i, ok in enumerate(results):
            f.write(f"ep{i}\t{'success' if ok else 'fail'}\n")
        f.write(f"total\t{n_ok}/{len(results)}\n")


if __name__ == "__main__":
    main()

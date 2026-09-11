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
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------
# EDIT THESE
# ----------------------------------------------------------------------------
PY_SIM = "/home/cobot/simenv/opt/conda/envs/rlbench/bin/python"
COPPELIASIM_ROOT = "/home/cobot/simenv/content/CoppeliaSim"
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
    # Deploy allocates four large models back to back; expandable segments
    # avoids losing a few hundred MB to fragmentation between them.
    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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


SERVER_READY_MARKER = "waiting for client connection"


def wait_for_server(server_log, proc, timeout=180):
    """Wait for server.py by watching its log, NOT by connecting to the port.

    A TCP probe cannot be used here: server.py accepts exactly ONE client. A
    connect_ex() readiness check is accepted as that client, and when the probe
    closes the socket the server proceeds through `initial reset...` and dies
    with BrokenPipeError trying to send its init reply -- after which the real
    client gets ConnectionRefusedError. The probe consumed the accept().

    Watching the log also catches the server exiting during startup, which a
    port probe silently waits out until timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False              # server died before becoming ready
        try:
            with open(server_log, errors="replace") as f:
                if SERVER_READY_MARKER in f.read():
                    return True
        except FileNotFoundError:
            pass
        time.sleep(2)
    return False


def port_free(port=PORT, host="127.0.0.1"):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port)); return True
        except OSError:
            return False


def kill_stale(timeout=60):
    subprocess.run("pkill -f coppeliaSim; pkill -f rlbench_tools/server.py",
                   shell=True, stderr=subprocess.DEVNULL)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_free():
            return True
        time.sleep(2)
    print(f"  WARNING: port {PORT} still bound after {timeout}s", flush=True)
    return False


REFERENCE_ROBOT = "sawyer"   # embodiment providing the one-shot demo


def discover_scenes(task, scenes_root, val_root):
    """Recovered scenes paired with the reference demo recorded IN that scene.

    Without this the simulator reset draws a fresh random scene while the demo
    comes from a specific episode, so the policy chases a target that is not
    where the demo says it is -- an unwinnable evaluation that looks like a
    policy failure. Scenes come from server/recover_rng.py.
    """
    d = os.path.join(scenes_root, task)
    if not os.path.isdir(d):
        sys.exit(f"ERROR: no recovered scenes at {d}\n"
                 f"Run:  server/recover_rng.py --task {task}")

    scenes = []
    for name in sorted(os.listdir(d)):
        scene_dir = os.path.join(d, name)
        if not os.path.isfile(os.path.join(scene_dir, "rng_state.pkl")):
            continue
        ref = os.path.join(val_root, task, REFERENCE_ROBOT, name + ".hdf5")
        if not os.path.isfile(ref):
            print(f"  WARNING: scene {name} has no {REFERENCE_ROBOT} demo "
                  f"({ref}); skipping", flush=True)
            continue
        scenes.append((name, scene_dir, ref))

    if not scenes:
        sys.exit(f"ERROR: {d} holds no usable scenes (need rng_state.pkl plus a "
                 f"matching {REFERENCE_ROBOT} demo under {val_root})")
    return scenes


def write_episode_config(ep, reference_h5, out_dir):
    """Per-episode copy of the deploy config, pointing at THIS episode's demo.

    Written per episode rather than mutating the committed config in place, so
    an interrupted run cannot leave a half-edited config behind.
    """
    import yaml
    c = yaml.safe_load(open(os.path.join(REPO, DEPLOY_CFG)))
    c["deploy"]["reference_h5"] = reference_h5
    path = os.path.join(out_dir, f"deploy_ep{ep}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(c, f, sort_keys=False)
    return path


def run_episode(ep, scene, args):
    scene_name, scene_dir, reference_h5 = scene
    frames_dir = os.path.join(args.out, f"ep{ep}")
    os.makedirs(frames_dir, exist_ok=True)
    server_log = os.path.join(args.out, f"server_ep{ep}.log")

    kill_stale()
    ensure_xvfb()

    # --episode_dir restores the scene the demo was recorded in. server.py also
    # reads the variation from that folder's meta.json, so it is not passed here.
    cmd = [
        PY_SIM, "-u", "server.py",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--task", args.task, "--robot", args.robot,
        "--episode_dir", scene_dir,
        "--image_size", "256", "256",
        "--renderer", "opengl", "--headless",
        "--save_image_dir", frames_dir,
    ]
    with open(server_log, "w") as log:
        srv = subprocess.Popen(cmd, cwd=os.path.join(REPO, "scripts", "rlbench_tools"),
                               env=sim_env(), stdout=log, stderr=subprocess.STDOUT)

    print(f"  simulator starting (CoppeliaSim, up to {args.timeout}s)...", flush=True)
    if not wait_for_server(server_log, srv, timeout=args.timeout):
        why = ("server exited during startup" if srv.poll() is not None
               else f"server never became ready within {args.timeout}s")
        srv.terminate()
        print(f"  ERROR: {why}. See {server_log}", flush=True)
        return False, 0

    # Report the MPC settings actually in the config, not a hardcoded guess --
    # a stale literal here makes a reduced-compute run look like a paper run.
    ep_cfg = write_episode_config(ep, reference_h5, args.out)
    try:
        import yaml
        _m = yaml.safe_load(open(ep_cfg))["deploy"]["mpc"]
        _s, _c = _m.get("samples"), _m.get("cem_steps")
        _note = "" if (_s, _c) == (200, 50) else "  *** REDUCED, not reportable ***"
        print(f"  scene {scene_name}  demo {os.path.basename(reference_h5)}", flush=True)
        print(f"  running policy (CEM: {_s} samples x {_c} steps per env step){_note}",
              flush=True)
    except Exception as _e:
        print(f"  running policy (could not read mpc from {ep_cfg}: {_e})", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "app.vjepa_2_1_dreamer_ac.deploy", "--fname", ep_cfg],
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

    n_frames = sum(1 for _dp,_dn,files in os.walk(frames_dir)
                   for f in files if f.lower().endswith((".png",".jpg")))

    # Always written, not just on failure. deploy.py prints the per-step chosen
    # action and `latent l1 dist` to stdout, which is the only record of whether
    # the planner is saturating and whether the subgoal threshold is gating --
    # i.e. exactly what a SUCCESSFUL-looking run needs to be judged on. Keeping
    # it only for crashes threw away every useful diagnostic.
    err_log = os.path.join(args.out, f"deploy_ep{ep}.log")
    with open(err_log, "w", errors="replace") as f:
        f.write("===== STDOUT =====\n" + (proc.stdout or "")
                + "\n===== STDERR =====\n" + (proc.stderr or ""))

    if proc.returncode != 0:
        # stderr matters most: Python tracebacks go there, not to stdout, so
        # printing only stdout showed model-init logging and hid the exception.
        err = "\n".join((proc.stderr or "").splitlines()[-25:])
        out = "\n".join((proc.stdout or "").splitlines()[-8:])
        print(f"  deploy.py exited {proc.returncode}", flush=True)
        if err.strip():
            print(f"  --- stderr (last 25) ---\n{err}", flush=True)
        else:
            print(f"  --- stdout (last 8, stderr empty) ---\n{out}", flush=True)
        print(f"  full output: {err_log}", flush=True)

    return success, n_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=0,
                   help="0 = one rollout per recovered scene (the usual case)")
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--robot", type=str, default="panda", help="target embodiment")
    p.add_argument("--out", type=str, default=os.path.join(REPO, "rollouts"))
    p.add_argument("--scenes", type=str, default=os.path.join(REPO, "data", "scenes"),
                   help="output of server/recover_rng.py")
    p.add_argument("--val-root", type=str, default=os.path.join(REPO, "data", "val"),
                   help="holds the reference demos, one per scene")
    p.add_argument("--timeout", type=int, default=180, help="seconds to wait for CoppeliaSim")
    p.add_argument("--fresh", action="store_true", help="wipe --out first")
    args = p.parse_args()

    if args.fresh and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(PY_SIM):
        sys.exit(f"ERROR: PY_SIM not found: {PY_SIM}\nEdit the path at the top of this file.")

    scenes = discover_scenes(args.task, args.scenes, args.val_root)

    n = args.episodes if args.episodes > 0 else len(scenes)
    if n > len(scenes):
        # Reusing a scene would re-measure the same pairing and inflate the
        # denominator with duplicates, so clamp rather than wrap around.
        print(f"NOTE: asked for {n} episodes but only {len(scenes)} scenes are "
              f"recovered; running {len(scenes)}.")
        n = len(scenes)

    print(f"xvfb: {ensure_xvfb()}   task: {args.task}   episodes: {n}   "
          f"scenes: {len(scenes)}")

    results = []
    for ep in range(n):
        print(f"\n=== episode {ep + 1}/{n} ===", flush=True)
        ok, n_frames = run_episode(ep, scenes[ep], args)
        results.append((scenes[ep][0], ok))
        print(f"  -> {'SUCCESS' if ok else 'FAIL'}   ({n_frames} frames)", flush=True)

    n_ok = sum(ok for _name, ok in results)
    print(f"\n{'=' * 46}")
    print(f"SUCCESS RATE: {n_ok}/{len(results)} = {100 * n_ok / max(len(results), 1):.1f}%")
    print(f"frames under: {args.out}")
    print(f"{'=' * 46}")

    with open(os.path.join(args.out, "results.txt"), "w") as f:
        for i, (name, ok) in enumerate(results):
            f.write(f"ep{i}\t{name}\t{'success' if ok else 'fail'}\n")
        f.write(f"total\t{n_ok}/{len(results)}\n")


if __name__ == "__main__":
    main()

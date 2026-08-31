#!/usr/bin/env python3
"""Measure peak VRAM and throughput per batch size, for stage 1 and stage 2.

Answers the only question that decides the lab-run config:
**what is the largest batch size that fits the 32GB lab card, and how fast is it?**

Two memory numbers are reported and they are NOT interchangeable:

  torch_alloc  torch.cuda.max_memory_allocated() -- PyTorch tensors only.
               This is what train.py logs as `mem`.
  nvidia_smi   true process VRAM, sampled from nvidia-smi while the run is live.
               Includes the CUDA context (~0.3-0.5GB), cuDNN workspaces and the
               caching allocator's reserved-but-unused blocks.

`nvidia_smi` is the one that decides whether a config fits a card. `torch_alloc`
is typically 1-3GB lower, and planning against it will OOM you.

Only ~10 optimizer steps are needed, NOT a full epoch -- but more than one step
IS required: Adam allocates exp_avg/exp_avg_sq lazily on the first
optimizer.step(), which is where stage 1 OOMed on the 16GB box. Peak is reached
shortly after step 1, so this runs `ipe` steps (default 10) and stops.

OOM is a RESULT, not a failure: it establishes the ceiling. The sweep records it
and continues.

Usage:
  python server/sweep_memory.py                          # stage 1, batches 8 16 32
  python server/sweep_memory.py --batches 8 16 32 48
  python server/sweep_memory.py --stages 1 2
  python server/sweep_memory.py --fits 32                # flag what fits 32GB
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGS = {
    1: "configs/train/vjepa_2_1_dreamer_predictor.yaml",
    2: "configs/train/vjepa_2_1_dreamer_ac.yaml",
}


def poll_gpu_mem(pid, stop, out):
    """Sample this PID's VRAM from nvidia-smi until told to stop. Records the max."""
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            for line in r.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] == str(pid):
                    out["peak"] = max(out.get("peak", 0), int(parts[1]))
        except Exception:
            pass
        time.sleep(0.5)


def run_one(stage, batch, ipe, env):
    """One short run. Returns a dict of results, or {'oom': True}."""
    rel = CFGS[stage]

    # prepare_configs.py reads its baseline from git HEAD and rewrites the file,
    # so set_batch.py must run AFTER it, never before. We only call set_batch
    # here -- the caller is responsible for having run prepare_configs once.
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "server", "set_batch.py"),
         "--batch", str(batch), "--stage", str(stage), "--measure"],
        cwd=REPO, capture_output=True, text=True,
    )
    if r.returncode != 0:
        return {"skipped": (r.stdout + r.stderr).strip().splitlines()[-1]}

    # Override ipe: --measure sets 50, but 10 steps is enough to pass Adam's
    # lazy state allocation and reach peak.
    import yaml
    p = os.path.join(REPO, rel)
    c = yaml.safe_load(open(p))
    c["optimization"]["ipe"] = ipe
    yaml.safe_dump(c, open(p, "w"), sort_keys=False)

    log_path = os.path.join(REPO, f"sweep_s{stage}_b{batch}.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.main", "--fname", rel,
             "--devices", "cuda:0", "--debugmode", "True"],
            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        smi = {}
        stop = threading.Event()
        t = threading.Thread(target=poll_gpu_mem, args=(proc.pid, stop, smi), daemon=True)
        t.start()
        rc = proc.wait()
        stop.set()
        t.join(timeout=3)

    text = open(log_path, errors="replace").read()
    if "OutOfMemoryError" in text or "CUDA out of memory" in text:
        return {"oom": True, "nvidia_smi": smi.get("peak"), "log": log_path}
    if rc != 0:
        tail = "\n".join(text.splitlines()[-3:])
        return {"error": f"exit {rc}: {tail}", "log": log_path}

    # Take the LAST log_stats line -- step 0 includes allocator warmup and
    # cuDNN autotune, which are not steady state.
    rows = re.findall(
        r"loss: ([\d.]+).*?\[mem: ([\d.e+]+)\].*?\[iter: ([\d.]+) ms\].*?\[gpu: ([\d.]+) ms\].*?\[data: ([\d.]+) ms\]",
        text,
    )
    if not rows:
        return {"error": "no log_stats lines parsed", "log": log_path}
    loss, mem, it, gpu, data = rows[-1]
    return {
        "loss": float(loss),
        "torch_alloc": float(mem),          # MB
        "nvidia_smi": smi.get("peak"),      # MB
        "iter_ms": float(it),
        "gpu_ms": float(gpu),
        "data_ms": float(data),
        "ms_per_sample": float(gpu) / batch,
        "log": log_path,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--stages", type=int, nargs="+", default=[1])
    p.add_argument("--ipe", type=int, default=10,
                   help="optimizer steps per config; >1 required for Adam state")
    p.add_argument("--fits", type=int, default=32,
                   help="flag configs fitting this many GB (the lab card)")
    args = p.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("WANDB_MODE", "disabled")
    # NOTE: expandable_segments is deliberately NOT set. It reduces fragmentation,
    # which would make these numbers look better than a default-configured run.
    # Measure the pessimistic case; enable it for real runs if you want the margin.

    results = {}
    for stage in args.stages:
        for batch in args.batches:
            print(f"\n=== stage {stage}, batch {batch} ===", flush=True)
            r = run_one(stage, batch, args.ipe, env)
            results[(stage, batch)] = r
            if "skipped" in r:
                print(f"  SKIPPED: {r['skipped']}", flush=True)
            elif "oom" in r:
                print(f"  OOM (ceiling found)", flush=True)
            elif "error" in r:
                print(f"  ERROR: {r['error']}", flush=True)
            else:
                print(f"  torch_alloc {r['torch_alloc']/1024:.1f} GB | "
                      f"nvidia-smi {(r['nvidia_smi'] or 0)/1024:.1f} GB | "
                      f"gpu {r['gpu_ms']:.0f} ms | "
                      f"{r['ms_per_sample']:.1f} ms/sample", flush=True)

    budget_mb = args.fits * 1024
    print("\n" + "=" * 78)
    print(f"{'stage':>5} {'batch':>5} {'torch GB':>9} {'smi GB':>8} "
          f"{'gpu ms':>8} {'ms/sample':>10} {'samp/s':>8}  fits {args.fits}GB?")
    print("-" * 78)
    for (stage, batch), r in results.items():
        if "oom" in r:
            print(f"{stage:>5} {batch:>5} {'OOM':>9}")
            continue
        if "skipped" in r or "error" in r:
            print(f"{stage:>5} {batch:>5} {'--':>9}  {r.get('skipped') or r.get('error')}")
            continue
        smi = r["nvidia_smi"]
        fits = "?" if smi is None else ("YES" if smi < budget_mb * 0.92 else "no")
        print(f"{stage:>5} {batch:>5} {r['torch_alloc']/1024:>9.1f} "
              f"{(smi or 0)/1024:>8.1f} {r['gpu_ms']:>8.0f} "
              f"{r['ms_per_sample']:>10.1f} {1000/r['ms_per_sample']:>8.1f}  {fits}")
    print("=" * 78)
    print("fits = nvidia-smi peak under 92% of budget (headroom for fragmentation).")
    print("Plan the lab run on the largest YES. Use ms/sample, not gpu ms, to")
    print("compare batch sizes -- gpu ms grows with batch by construction.")
    print("\nWARNING: nvidia-smi is sampled at 2 Hz. A brief spike between samples")
    print("can be missed, so treat these as lower bounds and keep real headroom.")


if __name__ == "__main__":
    main()

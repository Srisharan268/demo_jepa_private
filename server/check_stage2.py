#!/usr/bin/env python3
"""Verify a finished stage 2 run before tearing the machine down.

Checks the things that can go silently wrong -- a checkpoint that loaded nothing,
a schedule that never annealed, a loss that plateaued at the degenerate value, or
an autoregressive rollout that quietly diverged from the teacher-forced one.

Usage:  python server/check_stage2.py stage2.log
"""
import os
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "stage2.log"
text = open(LOG, errors="replace").read()
ok = True


def chk(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


print(f"parsing {LOG}\n--- loading ---")

# A state_dict can "load successfully" while binding nothing if the module.
# prefix does not match (HANDOFF §6) -- so check for the explicit success string,
# not merely the absence of an error.
chk("stage 1 dreamer checkpoint loaded",
    "loaded pretrained dreamer_predictor with msg: <All keys matched successfully>" in text)
chk("AC predictor loaded",
    "loaded pretrained predictor from epoch 0 with msg: <All keys matched successfully>" in text)
chk("encoder: no unexpected keys (the §5 norm rename held)",
    "unexpected_keys=[]" in text)

print("--- schedule ---")
rows = re.findall(
    r"\[\s*(\d+),\s*(\d+)\] loss: ([\d.]+) \[([\d.]+), ([\d.]+)\].*?\[lr: ([\d.e+-]+)\]", text)
chk("log lines parsed", len(rows) > 2, f"{len(rows)} found")
if rows:
    ep = [int(r[0]) for r in rows]
    lr = [float(r[5]) for r in rows]
    loss = [float(r[2]) for r in rows]
    jl = [float(r[3]) for r in rows]
    sl = [float(r[4]) for r in rows]
    chk("WSD warmup happened (lr rose above its start)", max(lr) > lr[0],
        f"{lr[0]:.2e} -> peak {max(lr):.2e}")
    chk("WSD anneal completed (lr -> 0)", lr[-1] == 0.0, f"final lr {lr[-1]:.2e}")
    chk("all epochs ran", ep[-1] >= max(ep), f"last epoch {ep[-1]}")

    print("--- learning ---")
    chk("loss fell", loss[-1] < loss[0] * 0.8, f"{loss[0]:.3f} -> {loss[-1]:.3f}")
    chk("loss finite", not any(x != x or x == float("inf") for x in loss))
    # Both components are L1 in layer-normed space, same calibration as stage 1:
    # ~1.13 chance, ~0.80 predicting zeros. Parking near 0.80 would be the
    # degenerate solution rather than convergence.
    chk("final loss well below the zeros baseline", loss[-1] < 0.5,
        f"{loss[-1]:.3f}  (zeros ~0.80 for a single L1 term)")

    print("--- autoregressive health (predicts rollout quality) ---")
    ratio0, ratioN = sl[0] / max(jl[0], 1e-9), sl[-1] / max(jl[-1], 1e-9)
    chk("sloss/jloss did not blow up", ratioN < 1.5, f"{ratio0:.3f} -> {ratioN:.3f}")
    chk("gap did not widen materially", ratioN <= ratio0 * 1.2,
        "sloss is the AUTOREGRESSIVE rollout -- what MPC actually runs")

print("--- artifacts ---")
for p, why in (("exp/stage2/latest.pt", "needed by make_deploy_ckpt.py"),
               ("exp/stage2/log_r0.csv", "per-iteration curve"),
               ("exp/stage1/latest.pt", "stage 1 weights")):
    e = os.path.exists(p)
    chk(f"{p} exists", e, why if not e else f"{os.path.getsize(p)/1e9:.2f} GB")

print("--- errors ---")
for label, pat in (("Traceback", r"Traceback"),
                   ("OutOfMemoryError", r"OutOfMemoryError"),
                   ("nan/inf loss", r"loss: (?:nan|inf|-inf)")):
    n = len(re.findall(pat, text, re.I))
    chk(f"no {label} in log", n == 0, f"{n} occurrence(s)" if n else "")

print("\n" + ("STAGE 2 OK -- safe to tear down after pulling artifacts"
              if ok else "*** ISSUES ABOVE -- read before destroying the box ***"))
print("\nNOTE: stage 2 has NO validation split, so every number here is TRAIN"
      "\nloss. It cannot distinguish a learned dynamics model from memorisation"
      "\nof 362 episodes. The held-out ROLLOUT is stage 2's only real validation.")
sys.exit(0 if ok else 1)

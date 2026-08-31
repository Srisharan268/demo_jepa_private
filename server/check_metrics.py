#!/usr/bin/env python3
"""Verify the Tier 1/2 metrics are wired correctly, BEFORE spending GPU hours.

Two parts:

  --offline   no GPU, no checkpoint. Feeds synthetic latents through the exact
              expressions train.py uses and asserts they respond correctly to
              known-good, known-collapsed and known-random inputs. Proves the
              MATH is right.

  --live      parses a real training log and asserts every metric actually
              appeared and moved. Proves the WIRING is right -- a metric that
              is computed but never logged, or logged as a constant, is worse
              than none at all because it looks like evidence.

Usage:
  python server/check_metrics.py --offline
  python server/check_metrics.py --live stage1_smoke.log
"""
import argparse
import re
import sys

import torch
import torch.nn.functional as F


def offline():
    torch.manual_seed(0)
    B, T, D = 8, 512, 1408
    ok = True

    def check(name, got, lo, hi):
        nonlocal ok
        good = lo <= got <= hi
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:38s} {got:8.4f}  expect {lo}-{hi}")

    tgt = F.layer_norm(torch.randn(B, T, D), (D,))

    print("L1 baselines (these calibrate what a loss value MEANS):")
    check("unrelated latent  == chance",
          float(F.l1_loss(F.layer_norm(torch.randn(B, T, D), (D,)), tgt)), 1.05, 1.20)
    check("all zeros == degenerate soln",
          float(F.l1_loss(torch.zeros_like(tgt), tgt)), 0.75, 0.85)
    check("perfect prediction", float(F.l1_loss(tgt, tgt)), 0.0, 0.001)

    print("\ncollapse monitor (std) -- must separate collapse from convergence:")
    good = tgt + 0.3 * torch.randn_like(tgt)
    coll = tgt.mean(dim=0, keepdim=True).expand_as(tgt).contiguous()
    check("healthy model std", float(good.float().std(dim=0).mean()), 0.5, 2.0)
    check("COLLAPSED model std", float(coll.float().std(dim=0).mean()), 0.0, 1e-4)
    # The point of the metric: a collapsed model beats the zeros baseline on L1.
    check("collapsed L1 (BELOW zeros baseline!)", float(F.l1_loss(coll, tgt)), 0.60, 0.80)

    print("\ncosine similarity:")
    check("healthy", float(F.cosine_similarity(good.flatten(1), tgt.flatten(1), -1).mean()), 0.8, 1.0)
    check("random", float(F.cosine_similarity(
        torch.randn_like(tgt).flatten(1), tgt.flatten(1), -1).mean()), -0.1, 0.1)

    print("\nretrieval (same expression as train.py::_retrieval):")

    def retr(p, t):
        p, t = F.normalize(p, dim=-1), F.normalize(t, dim=-1)
        n = p.size(0)
        topk = (p @ t.T).topk(min(5, n), dim=-1).indices
        return float((topk[:, 0] == torch.arange(n)).float().mean())

    pool_t = tgt.mean(1)
    check("perfect -> top1 1.0", retr(pool_t.clone(), pool_t), 0.99, 1.0)
    check("random  -> top1 ~chance", retr(torch.randn_like(pool_t), pool_t), 0.0, 0.30)

    print("\n" + ("ALL OFFLINE CHECKS PASSED" if ok else "*** FAILURES ABOVE ***"))
    return 0 if ok else 1


def live(path):
    text = open(path, errors="replace").read()
    ok = True

    def need(name, pat, minhits=1):
        nonlocal ok
        hits = re.findall(pat, text)
        good = len(hits) >= minhits
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:34s} {len(hits)} occurrence(s)")
        return hits

    print(f"parsing {path}\n")
    need("validation loader created", r"validation loader:")
    std = need("std logged", r"\[std: ([\d.]+)\]", 2)
    cos = need("cos logged", r"\[cos: ([-\d.]+)\]", 2)
    need("samp/s logged", r"\[samp/s: ([\d.]+)\]", 2)
    vals = need("VAL line", r"VAL ([\d.]+)")
    need("retrieval line", r"retrieval top1 ([\d.]+)")

    # A metric that never changes is not being computed -- it is a constant
    # being printed, which is the failure mode that looks most like success.
    print()
    for name, vs in (("std", std), ("cos", cos)):
        if len(vs) >= 2:
            f = [float(v) for v in vs]
            moved = len(set(f)) > 1
            ok &= moved
            print(f"  [{'PASS' if moved else 'FAIL'}] {name} varies across steps "
                  f"(min {min(f):.3f} max {max(f):.3f})")

    if std:
        last = float(std[-1])
        if last < 0.01:
            print(f"\n  *** WARNING: std={last:.4f} -- representation COLLAPSE. "
                  f"The model is emitting a near-constant.")
    if vals:
        v = float(vals[-1])
        verdict = ("above chance -- not learning" if v > 1.05 else
                   "at/above the zeros baseline -- suspect" if v > 0.78 else
                   "below the zeros baseline -- predicting real structure")
        print(f"\n  last VAL {v:.4f}: {verdict}")

    print("\n" + ("ALL LIVE CHECKS PASSED" if ok else "*** FAILURES ABOVE ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true")
    p.add_argument("--live", metavar="LOG")
    a = p.parse_args()
    if a.live:
        sys.exit(live(a.live))
    sys.exit(offline())

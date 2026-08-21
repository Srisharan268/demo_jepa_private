# Demo-JEPA project handoff

Everything a fresh session needs. Read this fully before running or changing anything.

---

## 1. Who and what

Srisharan, an intern, is reproducing **Demo-JEPA** (arXiv 2605.20811) — "Joint-Embedding
Predictive Architecture for One-shot Cross-Embodiment Imitation" — at reduced scale
on limited GPU. Repo: `banban3forever/Demo-JEPA`, upstream commit `a864863`
(verified byte-identical to upstream via live `git ls-remote`).

**The method.** A reference demonstration by robot A (sawyer) is translated into
latent goals; robot B (franka) executes via CEM/MPC planning over a learned
action-conditioned world model.

Working dir: `C:\WSAIS Intern\Demo-JEPA` (Windows laptop).
Branch `server-4gpu`. Two remotes: `origin` = upstream (read-only reference),
`lab` = `Srisharan268/demo_jepa_private` (private, push here).

**User context that matters:** they lean on the assistant heavily for technical
decisions, but their instincts have been repeatedly right — they caught a
`diffusers` red herring, challenged a wrong claim about world-model fidelity, and
pushed back on an over-engineered automation harness (which was then deleted).
Take their pushback seriously. They asked explicitly for **brutal honesty** about
feasibility. Give it.

---

## 2. Scope — settled decisions, do not relitigate

| Decision | Status |
|---|---|
| **Stage 0 skipped** — load Meta's released V-JEPA 2.1-AC | settled |
| **Imitation stage dropped** — it is a Diffusion Policy baseline, README labels it "Extra" | settled |
| `diffusion_policy` submodule NOT needed | settled |
| `diffusers` NOT needed — only imported by the imitation stage, not in requirements.txt | settled |
| Run **stages 1 and 2 only** | settled |
| Sim embodiments: **sawyer (source) → franka (target)** — the paper's sim setup | settled |

---

## 3. Hardware — CHANGED SEVERAL TIMES, this is current

| Machine | Spec | Role |
|---|---|---|
| **Lab final** | **1× RTX PRO 4500 Blackwell, 32GB** | the real run. Originally 4 GPUs — now ONE. |
| **Lab test box** | RTX 4080 Super 16GB, 125GB RAM, 32 cores, 403GB free, driver 570.211.01 (CUDA 12.8) | smoke testing, free, available now |
| Laptop | Windows, cmd.exe | edit + git only. No rsync (use `scp`). |

Budget: **4–5 days GPU total on the lab card, no redos.** Data generation in
simulation is effectively unlimited.

Blackwell is `sm_120` → **bf16 is native**. The V100/float16 worry is dead;
`dtype: bfloat16` stays exactly as upstream.

---

## 4. THE CURRENT BLOCKER — unfixed

Stage 1 crashes immediately on the smoke run:

```
RuntimeError: expect_autograd_hooks_ INTERNAL ASSERT FAILED
  at "/pytorch/torch/csrc/distributed/c10d/reducer.cpp":1705
```

**Cause.** The gradient-accumulation code added to
`app/vjepa_2_1_dreamer_predictor/train.py` (~line 452) uses
`dreamer_predictor.no_sync()`. But line 290 wraps it as
`DistributedDataParallel(dreamer_predictor, static_graph=True)`, and PyTorch
documents `static_graph=True` as **incompatible with `no_sync()`** — `no_sync()`
skips `prepare_for_backward` while the static graph assumes it ran.

**Proposed fix (NOT APPLIED — user rejected the edit mid-application; re-propose
and get agreement before editing).** Delete the `no_sync()` usage: drop the
`_sync` variable and the `with _sync:` block, de-indent the body, and iterate
`for current_frame, target_frame, current_reference, target_reference in micro_batches:`
without `enumerate`.

This is **mathematically identical**. DDP averages each micro-gradient across
ranks and they accumulate into `.grad`; averaging is linear, so
`avg(g1) + avg(g2) == avg(g1 + g2)`. The only cost is `n_micro - 1` redundant
all-reduces per step — **exactly zero on a single GPU**, which is what the final
run now uses anyway.

Alternative if you prefer: remove `static_graph=True` from the DDP wrap instead.
More invasive (changes upstream behavior); not recommended.

---

## 5. SECOND UNRESOLVED ISSUE — encoder checkpoint key mismatch

Also visible in that same run, and **nobody has investigated it yet**:

```
loaded pretrained encoder with msg: _IncompatibleKeys(
  missing_keys=['module.img_mod_embed', 'module.video_mod_embed',
                'module.patch_embed_img.proj.weight', 'module.patch_embed_img.proj.bias',
                'module.norms_block.0.weight' ... 'module.norms_block.3.bias'],
  unexpected_keys=['module.norm.weight', 'module.norm.bias'])
```

The released V-JEPA 2-AC checkpoint has a single final `norm`; the model built
from this config expects `norms_block.0-3` (from `n_output_distillation=4`) plus
`patch_embed_img` and modality embeddings (`modality_embedding: true`).

**Why it matters:** those tensors are randomly initialised. If the forward path
touches them, the "frozen pretrained encoder" is emitting partly-random features,
which would silently cap everything downstream. **Investigate before any long run.**
Check which of those modules the forward actually uses at
`img_temporal_dim_size: 1`, `modality_embedding: true`. It may be benign (the
video path may bypass `patch_embed_img`) — but confirm, do not assume.

---

## 6. Verified technical findings — trust these, they were measured or read from source

**Dataset semantics (this one bites).**
`__len__` returns the number of paired **EPISODES**, not frames
(`dataset.py:222`), and the loader sets `drop_last=True` (`dataset.py:47`).
If `batch_size > episodes/world_size`, `len(loader) == 0` and training
**hangs forever on StopIteration with no error**. `prepare_configs.py` now
counts pairs and exits 1 rather than writing a config that would hang.
`__getitem__` samples a **random frame pair per draw**, so a 93-frame episode
yields ~4,300 distinct samples — dataset diversity is combinatorial in frames.

**Memory (measured via meta-tensor autograd saved-tensor hooks, params excluded).**

| | |
|---|---|
| Encoder `vit_giant_xformers` | 1,013,267,968 params = 3.77 GB fp32 |
| `DreamerPredictor` | 424,733,121 params = 1.58 GB fp32 |
| Stage 1 static (params+grads+Adam) | ~10.1 GB |
| Stage 1 activations | **1,042 MB/sample (pure bf16) to 1,867 MB/sample (fp32)**; real autocast sits near the top |
| Stage 1 @ batch 16 | **34–47 GB → does NOT fit 32GB.** batch 8 ≈ 24–27 GB (unvalidated on real HW) |

**Why activations are so large.** `Conv3dFusionNetwork` in
`src/models/dreamer_predictor.py` holds **150,721 params (0.04% of the model) but
58% of activation memory** — it inflates tokens to `(B, 64, 1408, 16, 16)`,
46 MB/sample per tensor, ~19 retained tensors, and **`DreamerPredictor` has no
activation checkpointing** (`use_activation_checkpointing` never reaches
`get_dreamer_predictor`).

**Trainable modules per stage** (encoder is frozen everywhere we run):
- Stage 1 → `dreamer_predictor` only (`unfreeze_vit` defaults False)
- Stage 2 → AC `predictor` only (`unfreeze_dreamer_predictor` defaults False)
- Both run the encoder inside `torch.no_grad()`

**Scheduler is WSD** (`src/utils/schedulers.py:9`): warmup → **flat** → anneal.
So `epochs` can be cut freely and you still get a properly annealed model.
`warmup`/`anneal` are **absolute** (do not scale them proportionally). Floor is
~40 epochs, below which the flat phase vanishes.

**Other verified facts.**
- `--devices` alone sets world size (`main.py:81`); `nodes`/`tasks_per_node` in the YAMLs are ignored on this path
- `--debugmode True` runs in-process, single GPU, no `mp.spawn` — this is why notebooks worked
- `load_checkpoint` always reports `start_epoch = 0` → **resuming restarts the LR schedule. Never stop/resume a run mid-flight.**
- `eval_freq: 100` is in all configs but **read by nothing** — there is no in-loop eval
- `CHECKPOINT_FREQ = 1` → `latest.pt` every epoch; `save_every_freq: 25` → `e{N}.pt`
- Three of `scripts/*.sh` point at `configs/train/vjepa_2_1/*.yaml`; real path is `configs/train/*.yaml` (upstream bug; `server/run_*.sh` use correct paths)
- `max_num_frames=512` builds a `torch.zeros(66048, 66048)` causal mask = **17.4 GB of HOST RAM** (not VRAM — `ac_predictor.py:156` slices before `.to(device)`). Optional fix in `server/optional_mask_size.py`
- Stage 0 checkpoint keys need renaming to `module.<name>` (strip `module.`/`backbone.`) or they load "successfully" while binding nothing — silent failure, handled by `server/repack_stage0.py`
- Checkpoint URL from `src/hub/backbones.py`: `https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`

---

## 7. The paper's actual results (fetched from arXiv HTML)

Simulation, Cross-Embodiment Bridging — **the suite matching our setting**:

| Task | VPP | XSkill | Demo-JEPA |
|---|---|---|---|
| Push Button | 0.63 | 0.53 | **0.60** |
| Pick and Lift | 0.27 | 0.07 | **0.43** |
| Slide Block to Target | 0.07 | 0.00 | **0.37** |
| Pick up Cup | 0.17 | 0.07 | **0.40** |

Zero-shot suite (held-out tasks): Put Rubbish in Bin 0.60, Phone on Base 0.45,
Close Drawer 0.33, Close Laptop Lid 0.20.

**The user once cited "80% on push_button" — that is `Press Button` in the
REAL-WORLD experiments, not simulation.** Sim push_button is 0.60, and VPP beats
Demo-JEPA on it.

**Scale (Table 1 / Table 14):** Stage 1 = **86 tasks, 13,444 trajectories**;
Stage 2 = 39 tasks, 8,324. Compute = **8×A100, 315 epochs**; stage 0 = 7 days,
stage 1 = 2.5 days, stage 2 = 1 day. Real-world used UR5e→Franka.

Calibration: 1 A100-day ≈ 4,700 stage-1 steps. RTX PRO 4500 ≈ 0.4–0.5× A100.

---

## 8. Honest feasibility assessment already delivered

Per-task training density is the one thing in the user's favour:
paper = 94,500 steps / 86 tasks ≈ 1,100 steps/task; a 4-task run at 16,500 steps
= 4,125 steps/task, i.e. **~3.7× more per task than the paper**.

Probabilities given (assuming they rehearse on rented/free hardware first):

| Outcome | Likelihood |
|---|---|
| Pipeline runs end to end, produces videos | 80% |
| Stage 1 retrieval clearly above chance | 85% |
| ≥1 task shows coherent purposeful motion | 70% |
| ≥1 task ≥30% success | ~50% |
| ≥1 task ≥50% success | ~25% |

**The deliverable framing was the key advice:** commit to *stage 1 retrieval
metrics (in-distribution + zero-shot) + qualitative rollout videos + an explicit
scale-comparison table* — ~80% likely — rather than "reproduce 0.60 on
push_button" — ~25% likely. Same experiment, different claim.

Recommended for a 4–5 day budget: 4 tasks × 600–800 pairs (narrow+deep, the
user's own instinct, and correct for maximising video quality), stage 1
`epochs: 55–60`, stage 2 `epochs: 45–55`, `ipe: 300`. Avoid grasping tasks for
headline numbers — prefer reach-and-actuate (`push_button`, `slide_block_to_target`,
`close_box`, `open_drawer`). Note the tension surfaced: Panda→UR5 is a stronger
claim but lower success; sawyer→franka is the paper's sim setting and safer.

Also flagged: **feature caching** (encoder is frozen, so its outputs are
precomputable) would give ~3× on stage 1, but needs ~1.7 TB at 80 tasks /
~270 GB with every-3rd-frame subsampling. Not implemented.

---

## 9. What has been built — `server/`

Small single-purpose scripts, called explicitly. An automation harness
(`smoketest.sh`) was built and then **deleted at the user's request** — do not
rebuild it. They want plain commands.

| File | Purpose |
|---|---|
| `HANDOFF.md` | this file |
| `SMOKE_RUNBOOK.md` | **14 plain numbered steps for the 16GB smoke test — the active document** |
| `RUNBOOK.md` | full 18-step lab runbook (written for 4 GPUs — **now stale on GPU count**) |
| `README.md` | deviations summary + order of operations |
| `repack_stage0.py` | 11GB → repacked; key renaming; asserts `module.` prefix |
| `split_dataset.py` | seeded (`Random(0)`) train/val split preserving episode pairing |
| `prepare_configs.py` | writes both configs; `--gpus N --smoke --epochs N`; **hang guard exits 1** |
| `patch_16gb.py` | notebook's bf16 patches for the 16GB box. **SMOKE ONLY — never merge** |
| `optional_mask_size.py` | `max_num_frames` 512→64, host-RAM fix |
| `run_stage1.sh` / `run_stage2.sh` | launchers (currently hardcode 4 devices — **need updating to 1 GPU**) |
| `run_eval_stage1.sh` | upstream retrieval eval; auto-sizes batch to held-out set |
| `make_deploy_ckpt.py` | stage 2 `latest.pt` → deploy checkpoint |
| `prepare_deploy_config.py` | deploy config; auto-selects reference demo; asserts paper MPC values |
| `run_rollout.py` | per-episode sim orchestration. **`PY_SIM`/`COPPELIASIM_ROOT` are the only real placeholders left** |
| `make_video.py` | side-by-side reference vs execution gif/mp4 |
| `install_sim_env.sh` / `use_prebuilt_sim.sh` | sim env from source (pinned) or from the user's 558MB prebuilt tarball |

**Code changes to upstream — exactly one file:**
`app/vjepa_2_1_dreamer_predictor/train.py` — gradient accumulation
(`accum_steps` config key, defaults to 1 = upstream behaviour). This is the file
with the bug in §4.

---

## 10. Data

In-repo: `data/rlbench_data.tar.gz` (38MB, committed — under GitHub's 100MB limit).
Extracts to `push_button/{franka,sawyer}/`, **18 paired episodes**, 93 frames each,
256×256, keys `observations/images/right_shoulder_rgb` + `observations/qpos`.
`split_dataset.py` → 12 train / 6 held-out.

**This is a smoke-test dataset, not a training set.** Real runs need collection
via `scripts/rlbench_tools/cli.py` — the paper's ratio is ~156 trajectories/task.

Sim env versions read out of the user's `rlbench_env.tar.gz`: **PyRep 4.1.0.3,
RLBench 1.2.0, CoppeliaSim 4.1 (Qt 5.12.5, boost 1.71 = Ubuntu 20.04 build),
Python 3.10, numpy 2.2.6, cffi 1.14.2, scipy 1.15.3, h5py 3.16.0.**
That tarball stores **absolute paths** (`opt/conda/envs/rlbench`,
`content/CoppeliaSim`) — extract to a user prefix, never `-C /`.

---

## 11. Immediate next steps

1. **Fix the `no_sync()` bug** (§4) — propose, get agreement, then edit
2. **Investigate the encoder key mismatch** (§5)
3. Re-run `SMOKE_RUNBOOK.md` step 7 on the 4080 box
4. Capture `[iter: ... ms]` and `[mem: ...]` — **the first real measurements in this project**; every time/memory number so far is arithmetic
5. Finish smoke steps 8–14 through to a gif
6. Then update `run_stage1.sh`/`run_stage2.sh` and `RUNBOOK.md` for **1 GPU**, and re-derive batch/accum

The user is on the 4080 box at `~/Demo-JEPA` (note: earlier path was
`~/Demo_JEPA/Demo-JEPA`), conda env `demojepa`, python 3.12,
torch built for cu130 originally — **must be cu128** to match driver 570.211.01.

---

## 12. Mistakes made in the prior session — avoid repeating

- Told them `pip install diffusers` was needed. It is not — imitation-only. **They caught it.**
- Claimed the world model was the weak link. Wrong: the AC predictor is Meta-pretrained; the **Dreamer Predictor (trained from scratch) is the weak link.** They caught it.
- Estimated stage 1 at "~9 days" from FLOP math; the paper's Table 14 implies ~4× more. **Prefer measurement over arithmetic.**
- Framed "9 days for any results" — wrong, WSD means any budget ≥40 epochs gives a real model.
- Said "data is free" re: training time. Half true — per-step cost is fixed, but required step count rises with data.
- Built an over-engineered `smoketest.sh` that introduced its own failures (wget flooding 220k lines through `tee`; a `[ -s "$f" ]` check that would accept a truncated download). **Deleted at their request. Do not rebuild.**
- Wrote `no_sync()` without checking `static_graph=True` compatibility — the current blocker.

**Working style they want:** verify against the actual source before asserting;
plain commands over automation; honest probabilities over reassurance; say
plainly when something cannot be de-risked.

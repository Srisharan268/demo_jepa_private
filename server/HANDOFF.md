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

**UPDATE 2026-08-22: there is NO deadline from the lab.** The project can run
as long as the user wants to invest. It is not yet established whether the
"4–5 days GPU" figure is a *card-availability* limit or was just the user's own
time budget — **find out, it is the single biggest lever in the project.** If
the card is available for longer, "no redos" stops being true and most of the
risk this document is built around disappears.

Blackwell is `sm_120` → **bf16 is native**. The V100/float16 worry is dead;
`dtype: bfloat16` stays exactly as upstream.

---

## 4. RESOLVED (2026-08-22) — the `no_sync()` / `static_graph` blocker

Stage 1 used to crash immediately with:

```
RuntimeError: expect_autograd_hooks_ INTERNAL ASSERT FAILED
  at "/pytorch/torch/csrc/distributed/c10d/reducer.cpp":1705
```

**Cause.** The gradient-accumulation code used `dreamer_predictor.no_sync()`
while the module is wrapped `DistributedDataParallel(..., static_graph=True)`.
PyTorch documents these as incompatible — `no_sync()` skips
`prepare_for_backward` while the static graph assumes it ran.

**Fixed** in commit `afef902`: `no_sync()` removed, DDP syncs on every
micro-batch. Mathematically identical (DDP averages each micro-gradient and
they accumulate into `.grad`; averaging is linear, so
`avg(g1) + avg(g2) == avg(g1 + g2)`), and the cost — `n_micro - 1` redundant
all-reduces — is **exactly zero on one GPU**, which is what we run.

**Verified:** stage 1 now runs 20 steps end to end and is deterministic.

---

## 5. RESOLVED (2026-08-22) — encoder checkpoint key mismatch

Traced through the source rather than assumed. Of the five missing-key
categories, **four are genuinely benign** and one was real:

| Missing key | Used in forward? | Verdict |
|---|---|---|
| `patch_embed_img.proj.*` | **No** | `dataset.py:209` does `np.repeat(images[:1], repeats=2)` → `T=2`. `check_temporal_dim` (`vision_transformer.py:272`) tests `shape[2] == img_temporal_dim_size` i.e. `2 == 1` → False → **video branch always**. The img branch is dead code here. |
| `img_mod_embed` | **No** | img branch only |
| `video_mod_embed` | Yes | `nn.init.normal_(std=1e-6)` — negligible, one broadcast vector |
| `norms_block.0-2` | Computed, **discarded** | appended to `hier`, which is unused when `training=False` and `return_hierarchical=False` (neither is ever set in stage 1). Wasted compute only. |
| `norms_block.3` | **YES — the output norm** | `vision_transformer.py:366`, `x = self.norms_block[-1](x)`. Was falling back to default init (weight=1, bias=0) while the checkpoint's trained `norm.*` was discarded. |

**Fixed** in commit `939124c`: `repack_stage0.py` now renames encoder
`norm.weight/bias` → `norms_block.3.weight/bias`. Encoder-only — the AC
predictor's final norm is `predictor_norm` (`ac_predictor.py:101`), untouched.
Index 3 because depth 40 → `hierarchical_layers [9,19,29,39]`, so
`norms_block[3]` sits after the last block, the position `norm` occupies.

**After the fix `unexpected_keys` is empty.** The remaining `missing_keys` are
the four benign categories above — expected, do not chase them.

Mattered most for **stage 2**, whose AC predictor is Meta-pretrained and expects
Meta's feature distribution. Stage 1 trains its dreamer from scratch and would
largely have absorbed the difference.

*Note: this mismatch is inherent to skipping stage 0 (§2). It is the gap between
"trained by stage 0" and "downloaded from Meta" made concrete — there may be
more of it. Worth re-checking against the V-JEPA 2-AC paper.*

---

## 5b. FIRST REAL RUNS (2026-08-22) — and why the timings are worthless

Stage 1 smoke, 4080 Super 16GB, `patch_16gb.py` applied (bf16 dreamer), batch 1
× accum 2, 20 steps:

| | step 0 | step 10 | step 19 |
|---|---|---|---|
| loss | 0.971 | 0.604 | 0.540 |
| mem | 8.75 GB | **9.30 GB** | 9.30 GB |
| gpu | 1800 ms | 731.9 ms | **685.5 ms** |

**Memory is trustworthy: 9.30 GB.** It independently corroborates §6's
meta-tensor analysis — add back fp32 for the dreamer (+3.4 GB) and 7 more
samples of activations (~1.0–1.9 GB each) and you land at ~23–24 GB for batch 8
fp32, inside §6's 24–27 GB estimate derived a completely different way.

**Timings are NOT trustworthy.** Three runs of identical compute gave
685 → 1069 → **1417 ms** GPU time, monotonically degrading. Cause: the box is
shared and **5.11 GB / significant CPU was held by another user's jobs**
(project `nero_four_cube_row`: two `check_actions.py`, one `src.train.bc`,
all under the same `cobot` account). Every timing sample this project has ever
taken was contended.

> **There is still NO valid throughput measurement for this project.** Every
> time estimate in §7/§8 and every extrapolation is arithmetic. Getting one
> uncontended `ms/sample` number, with a batch-size sweep and a repeat of the
> first batch size as a drift control, is the highest-value pending task.

Loss determinism confirmed: two runs gave bit-identical 0.971/0.604/0.540.
**Losses are only comparable when the encoder is unchanged** — the §5 fix
shifted the feature space, so pre-fix and post-fix losses cannot be compared.

**Stage 2 is blocked, not broken.** It OOMs at
`DistributedDataParallel(target_encoder)` (`train.py:336`) needing 1.89 GiB of
gradient buckets, with only ~10.5 GB available after the other users' 5.11 GB.
It got through model init and three other DDP wraps first. On 32 GB this will
not occur. Do not "fix" it — see §12.

**Deploy → rollout → video works end to end.** Proven 2026-08-22 using
`~/vjepa2_ac_repacked.pt` directly as the deploy checkpoint (it already has
`target_encoder` and `predictor`, the only two keys `make_deploy_ckpt.py`
needs), so stage 2 was not required. CoppeliaSim boots headless, frames land on
disk, `make_video.py` produces a gif. Motion is minimal — expected, with a
20-step dreamer, no stage-2 fine-tuning, and MPC cut to 20×5.

**wandb was uploading to a stranger's account.** `/home/cobot/.netrc` on the
shared box holds credentials for `thevishesh16` (IIT Kanpur). **Always set
`WANDB_MODE=disabled`** (`run_rollout.py` now defaults it). Do not delete the
netrc — it is not ours.

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
| `RUNBOOK.md` | **the rented-GPU measurement session — the active document on branch `cloud-test`** |
| *(`SMOKE_RUNBOOK.md`)* | 16GB smoke procedure. Finished; deleted on `cloud-test`, still on `server-4gpu` |
| `README.md` | deviations summary + order of operations |
| `repack_stage0.py` | 11GB → repacked; key renaming; asserts `module.` prefix |
| `split_dataset.py` | seeded (`Random(0)`) train/val split preserving episode pairing |
| `prepare_configs.py` | writes both configs; `--gpus N --smoke --epochs N`; **hang guard exits 1** |
| *(`patch_16gb.py`)* | bf16 patches for the 16GB box. **SMOKE ONLY.** Deleted on `cloud-test`; lives on `server-4gpu`/`test-16gb` |
| `optional_mask_size.py` | `max_num_frames` 512→64, host-RAM fix |
| `run_stage1.sh` / `run_stage2.sh` | launchers (currently hardcode 4 devices — **need updating to 1 GPU**) |
| `run_eval_stage1.sh` | upstream retrieval eval; auto-sizes batch to held-out set |
| `make_deploy_ckpt.py` | stage 2 `latest.pt` → deploy checkpoint |
| `prepare_deploy_config.py` | deploy config; auto-selects reference demo; asserts paper MPC values |
| `run_rollout.py` | per-episode sim orchestration. **`PY_SIM`/`COPPELIASIM_ROOT` are the only real placeholders left** |
| `make_video.py` | side-by-side reference vs execution gif/mp4 |
| `install_sim_env.sh` / `use_prebuilt_sim.sh` | sim env from source (pinned) or from the user's 558MB prebuilt tarball |

**Code changes to upstream — still exactly one file:**
`app/vjepa_2_1_dreamer_predictor/train.py` — gradient accumulation
(`accum_steps`, defaults to 1 = upstream behaviour), now with `no_sync()`
removed (§4). Everything else lives in `server/` or in `patch_16gb.py`, which
mutates files on a throwaway branch only.

**Sim environment (2026-08-22): installed and working.**
`~/simenv` on the 4080 box, from `use_prebuilt_sim.sh`. `PY_SIM` and
`COPPELIASIM_ROOT` in `run_rollout.py` are now set — §9's "only real
placeholders left" are gone. CoppeliaSim 4.1 boots headless under Xvfb :99;
every failed plugin is a GUI/ROS one and harmless (`IK`, `Vision`,
`OpenGL3Renderer`, all `Dynamics*` load fine).

*Gotcha:* the tarball stores **absolute symlink targets** (e.g.
`libcoppeliaSim.so.1 -> /content/CoppeliaSim/...`), not just absolute paths.
`use_prebuilt_sim.sh` now relocates them (commit `f86c767`). Re-running the
extractor restores the broken links, so re-run the script, never `tar` by hand.

**`~/.bashrc` on the SHARED `cobot` account** had `COPPELIASIM_ROOT`,
`LD_LIBRARY_PATH`, `QT_QPA_PLATFORM_PLUGIN_PATH` appended by that script.
Other users share this account. `run_rollout.py` sets all three itself, so the
`.bashrc` block is removable if it causes trouble (backup at `~/.bashrc.bak`).

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

## 11. Immediate next steps (rewritten 2026-08-22)

Smoke testing is essentially **done** — §4 and §5 closed, stage 1 trains,
sim works, rollout→gif proven. What remains:

1. **Get an uncontended throughput measurement.** The one true blocker on all
   planning. Needs the 4080 (or rented GPU) with no other jobs: verify via
   `nvidia-smi`, then sweep batch 1/2/4 and **re-run batch 1 last as a drift
   control**. Compare `ms/sample` = `gpu_ms / (batch_size × accum_steps)`.
2. **Establish whether the lab card's "4–5 days" is calendar or availability**
   (§3). Changes everything downstream.
3. **Run stage 2 to completion** — needs ~2 GB more than the shared 4080
   currently offers. Will not be a problem on 32 GB or on rented hardware.
4. **User is learning the codebase before spending money** — deliberate, agreed,
   and correct. Assist by checking their reasoning rather than supplying
   answers; they explicitly want to stop being a passenger.
5. Rehearsal run on **rented cloud GPU** before the lab card.
6. ~~Update `run_stage1.sh`/`run_stage2.sh` and `RUNBOOK.md` for 1 GPU~~ — done on `cloud-test`.
7. Data collection for real tasks via `scripts/rlbench_tools/cli.py` —
   **never timed; measure episodes/hour before planning around it.**

**Worth building if compute stays tight: feature caching.** The encoder is
frozen and runs under `no_grad()` in both stages, so its outputs are
precomputable. §8 estimates ~3× on stage 1 for ~270 GB with every-3rd-frame
subsampling; the test box has 403 GB free. Verify the 3× before committing —
it is arithmetic (see §12).

The user is on the 4080 box at `~/Demo-JEPA`, conda env `demojepa`, python 3.12,
torch **cu128** to match driver 570.211.01. Branch `test-16gb` there carries the
`patch_16gb.py` mutations; `server-4gpu` is clean; **`cloud-test` is the branch for
the rented-GPU session** (no 16GB hacks, 1 GPU, single `RUNBOOK.md`).

---

## 12. Mistakes made in the prior session — avoid repeating

- Told them `pip install diffusers` was needed. It is not — imitation-only. **They caught it.**
- Claimed the world model was the weak link. Wrong: the AC predictor is Meta-pretrained; the **Dreamer Predictor (trained from scratch) is the weak link.** They caught it.
- Estimated stage 1 at "~9 days" from FLOP math; the paper's Table 14 implies ~4× more. **Prefer measurement over arithmetic.**
- Framed "9 days for any results" — wrong, WSD means any budget ≥40 epochs gives a real model.
- Said "data is free" re: training time. Half true — per-step cost is fixed, but required step count rises with data.
- Built an over-engineered `smoketest.sh` that introduced its own failures (wget flooding 220k lines through `tee`; a `[ -s "$f" ]` check that would accept a truncated download). **Deleted at their request. Do not rebuild.**
- Wrote `no_sync()` without checking `static_graph=True` compatibility. Fixed 2026-08-22 (§4).

**Session of 2026-08-22 — all the same root cause: asserting a mechanism
instead of verifying it.**

- Proposed freezing `target_encoder` *before* its DDP wrap to skip gradient-bucket allocation. PyTorch rejects this outright: `DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient`. Reverted (`daea488`). Skipping the wrap is also unsafe — `_normalize_pretrained_keys` strips only `backbone.`, never `module.`, so an unwrapped `target_encoder` would `load_state_dict(strict=False)` and bind **nothing**, silently.
- Claimed GradScaler on bf16 was "harmless, just wasted time." It is a hard `NotImplementedError` — `_amp_foreach_non_finite_check_and_unscale_` has no BFloat16 kernel. That is *why* `patch_stage2` bypasses the scaler; the reasoning should have been carried across rather than re-derived from numerics.
- Placed deploy's bf16 casts *after* all four models were built, when the OOM happens *during* construction of the fourth. The casts never executed. The tell was memory coming back byte-identical (9.99 → 10.00 GB): **a patch that changes no measurable number did not run.**
- Inferred the MPC config from a hardcoded print string in `run_rollout.py` and told the user they had skipped a step they had not.

**A whole class of bug found in `server/` tooling: scripts trusting output they
themselves wrote.** `prepare_configs.py` and `prepare_deploy_config.py` both
read, validated, and overwrote the same file, so a second run validated its own
output — and in the deploy case reported a deliberate local MPC reduction as
"upstream changed". Both now read the baseline from `git show HEAD:<path>`.
Related: `run_rollout.py` printed hardcoded MPC values regardless of config, and
printed only `stdout` on failure while tracebacks go to `stderr`.
**If you add a script here, make it idempotent and make it report what is real.**

- `wait_for_port()` probed readiness with `connect_ex()`. `server.py` accepts exactly ONE client, so the probe *was* the client: the server accepted it, the probe closed the socket, and the server died with `BrokenPipeError` sending its init reply — after which `deploy.py` got `ConnectionRefusedError` from a different log. **The readiness check destroyed the thing it checked.** Now waits on the server log (`fda1072`).

**Working style they want:** verify against the actual source before asserting;
plain commands over automation; honest probabilities over reassurance; say
plainly when something cannot be de-risked.

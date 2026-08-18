# Demo-JEPA runbook — 4×32GB lab server

Stage 0 is skipped (pretrained V-JEPA 2.1-AC). Stages 1 and 2 only. The
imitation experiment is a baseline and is out of scope.

Each step has a **✓ check**. If a check fails, stop there — later steps assume it passed.

## Where each step runs

Two machines. **Step 1 is the only thing you do on your laptop** — everything
after it happens over SSH, in `~/Demo-JEPA` on the server.

| | |
|---|---|
| 🖥 **LAPTOP** | `C:\WSAIS Intern\Demo-JEPA` — step 1 only |
| ☁ **SERVER** | `~/Demo-JEPA` after step 3 — steps 2–18 |

---

# SECTION 1 — Get training running

## 1. 🖥 LAPTOP — push the code to a private repo

Your local folder is ~6.5 MB of pure code (no checkpoints, no datasets), so all
of it can go. `.gitignore` keeps future checkpoints and data out.

```bash
git checkout -b server-4gpu
```

```bash
git add .gitignore app/vjepa_2_1_dreamer_predictor/train.py server/ && git commit -m "Gradient accumulation for stage 1; server run scripts and runbook"
```

Create the repo at <https://github.com/new> — **Private**, and leave "Add a
README", "Add .gitignore", and "Choose a license" **unchecked**. An initialized
repo creates a commit your branch does not share, and the push is rejected as
non-fast-forward.

```bash
git remote add lab https://github.com/Srisharan268/demo_jepa_private.git
```

```bash
git push -u lab server-4gpu
```

**✓ check:** prints `* [new branch] server-4gpu -> server-4gpu`, and the repo
page shows a `server/` folder with 11 files.

> Notes:
> - `gh` CLI is not required; the web UI plus plain git is enough. If a remote
>   named `lab` already exists, use `git remote set-url lab <url>` rather than
>   `add`.
> - "Repository not found" on a **private** repo means *either* a wrong name
>   *or* failed auth — GitHub returns 404 instead of 403. If the page loads in
>   your browser, it is auth: create a token at
>   <https://github.com/settings/tokens> with `repo` scope and use it as the
>   password.
> - GitHub's Fork button cannot make a public repo private — that is why this
>   creates a fresh private repo rather than forking. `origin` still points at
>   upstream, so `git fetch origin` continues to work for checking drift.

You are done with your laptop. Everything below runs on the server.

## 2. ☁ SERVER — check the GPUs

```bash
ssh you@server
```

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

**✓ check:** 4 GPUs listed, ~32GB each.

**Decision point:** if the name contains **V100**, bf16 is unsupported in hardware.
You must set `dtype: float16` in both training configs — a real deviation to
document. Ampere or newer (A100/L40/RTX 40xx/50xx): nothing to change.

## 3. ☁ SERVER — check resources, then clone

```bash
free -g && df -h ~ && nproc
```

**✓ check:** ≥64GB RAM free, ≥200GB disk. Note the RAM figure — if under ~64GB,
lower `NUM_WORKERS` in step 8 and expect to need `optional_mask_size.py` at stage 2.

The work is on the `server-4gpu` branch, so clone that explicitly rather than
relying on whatever GitHub picked as the default:

```bash
git clone -b server-4gpu https://github.com/Srisharan268/demo_jepa_private.git ~/Demo-JEPA
```

**✓ check:** `ls ~/Demo-JEPA/server/` shows 11 files. **From here on, every
command assumes you are in `~/Demo-JEPA` on the server.**

## 4. Build the conda environment

```bash
conda create -n djepa python=3.12 -y && conda activate djepa
```

```bash
cd ~/Demo-JEPA && pip install .
```

**✓ check:** finishes without a resolver error.

> The main README also tells you to `pip install diffusers==0.11.1` and then
> hand-edit `dynamic_modules_utils.py` to remove a `cached_download` import.
> **Skip both.** `diffusers` is imported in exactly one file —
> `app/vjepa_2_1_imitation/diffusion_head.py` — which belongs to the imitation
> baseline you are not running. It is not in `requirements.txt` either, so
> `pip install .` will not pull it.

## 5. Verify the install

```bash
python -c "import torch, timm, einops, h5py, decord, transformers; print('imports OK', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

**✓ check:** prints `imports OK <version> cuda: True`. `False` means torch cannot
see the GPUs — stop and fix that before going further.

```bash
python -c "import app.vjepa_2_1_dreamer_predictor.train, app.vjepa_2_1_dreamer_ac.train; print('stage 1 + 2 modules import OK')"
```

**✓ check:** prints the confirmation. This is the real test — it exercises the
actual training modules, including the gradient-accumulation change, rather than
just third-party packages.

## 6. Get the Stage 0 checkpoint

Download it **on the server**, not via your laptop — it is a public file and the
lab's bandwidth will beat your home upload. The URL comes from the repo's own
`src/hub/backbones.py` (`VJEPA_BASE_URL` + `vit_ac_giant` → `vjepa2-ac-vitg`),
and `vit_giant_xformers` matches the `model_name` in both training configs.

```bash
cd ~ && wget --show-progress https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt
```

**✓ check:** ~11GB downloaded.

Then repack — this drops the optimizer state, aliases `target_encoder`, and
renames keys for DDP:

```bash
cd ~/Demo-JEPA && python server/repack_stage0.py ~/vjepa2-ac-vitg.pt ~/vjepa2_ac_repacked.pt
```

**✓ check:** prints `top-level keys: ['encoder', 'predictor', ...]`, nonzero
tensor counts, and a `sample key` beginning with `module.`. That prefix is the
one that matters — the training scripts wrap models in DDP, and without it the
state dict loads "successfully" while binding nothing, giving you a loss that
never moves rather than an error.

> The raw file has **no `target_encoder`** — only `encoder` and `predictor`. The
> script aliases it, since both `context_encoder_key` and `target_encoder_key`
> resolve to `target_encoder` in the configs.
>
> Dtype is preserved by default. The Colab run cast to bf16 to fit in ~12.7GB of
> RAM; you do not need that on a real node, and preserving fp32 keeps the
> initial predictor weights at full precision. Pass `--bf16` only if RAM is tight.

## 7. Get the dataset in place

The dataset archive ships **inside the repo** (`data/rlbench_data.tar.gz`, 38MB),
so the clone already brought it. Extract and split:

```bash
cd ~/Demo-JEPA/data && tar -xzf rlbench_data.tar.gz && cd ..
```

```bash
python server/split_dataset.py
```

**✓ check:** prints `push_button   train 12   held-out 6`, creating `data/train`
and `data/val`. Both keep the `<task>/<robot>/` layout the dataloader expects.
The split is seeded (`random.Random(0)`), so it is identical on every machine.

Extracted `.hdf5` files are gitignored — only the archive is versioned, and the
split is reproducible from it.

> **This is a smoke-test dataset, not a training set.** 18 episode pairs of one
> task, 93 frames each. The configs run 94,500 optimizer steps at global batch
> 128 — thousands of passes over ~1,700 frames. Use it to prove the pipeline
> works end to end, then collect properly (the main [README.md](../README.md)
> uses `--total_episodes 200`) before any run you intend to report. Collecting
> needs the simulator env, so do Section 2 steps 15–16 first.

## 8. Write the training configs

Paths are repo-relative, so there is normally **nothing to edit** — `DATASET`
resolves to `data/train`, outputs to `exp/stage1` and `exp/stage2`. The only
absolute path is `STAGE0_CKPT` (`~/vjepa2_ac_repacked.pt` from step 6); change
it only if you put the checkpoint elsewhere.

```bash
python server/prepare_configs.py
```

**✓ check:** every path prints `[ok]`, and the global batches read
`stage 1: ... = global 128` and `stage 2: ... = global 16`. Any `[MISSING]` — fix before continuing.

## 9. Launch Stage 1

```bash
tmux new -s djepa
```

```bash
cd ~/Demo-JEPA && bash server/run_stage1.sh 2>&1 | tee stage1.log
```

**✓ checks, in order:**

| When | What you should see |
|---|---|
| ~30s | `loaded pretrained` with matching keys — no `size mismatch` |
| ~60s | **No CUDA OOM.** If it OOMs, set `batch_size: 4` / `accum_steps: 8` and relaunch |
| first log line | `[mem: ...]` between roughly **24000 and 27000** (MB) |
| a few lines in | `loss:` decreasing, not `nan` or `inf` |
| 300 steps | `latest.pt` appears in your stage 1 output folder — the real milestone |

Detach with `ctrl-b d`. Reattach with `tmux attach -t djepa`.

**Do not stop and resume mid-run** — `load_checkpoint` always reports
`start_epoch = 0`, so resuming restarts the LR schedule.

## 10. Launch Stage 2 (after Stage 1 finishes)

```bash
cd ~/Demo-JEPA && bash server/run_stage2.sh 2>&1 | tee stage2.log
```

**✓ check:** prints the stage 1 checkpoint path, then `loaded pretrained predictor`
and `loaded pretrained dreamer_predictor` with all keys matched. Loss lines show
a `[a, b]` pair — teacher-forcing loss and autoregressive-rollout loss.

If it dies at startup with a host-memory error, that's the 17.4GB causal mask:

```bash
python server/optional_mask_size.py
```

---

# SECTION 2 — Evaluation and rollout

## 11. Stage 1 eval — retrieval accuracy

```bash
bash server/run_eval_stage1.sh data/val
```

**✓ check:** prints `acc@1` well above chance. The script prints chance for you
(`1/batch_size` = 0.0625 at batch 16). At or near chance means the Dreamer
Predictor didn't learn — investigate before spending time on rollouts.

## 12. Build the deploy checkpoint

```bash
python server/make_deploy_ckpt.py exp/stage2/latest.pt exp/stage2_deploy.pt
```

**✓ check:** prints the written size (~5–6GB). Errors here mean stage 2's
checkpoint lacks `target_encoder` or `predictor`.

## 13. Write the deploy config

Nothing to edit — paths are repo-relative and the one-shot reference demo is
auto-selected from the held-out split.

```bash
python server/prepare_deploy_config.py
```

**✓ check:** all three paths `[ok]`, the reference resolves to a file under
`data/val/<task>/sawyer/`, and mpc reads `samples: 200, cem_steps: 50, topk: 10`
— the paper's values.

## 14. Validate CEM without the simulator

Do this before touching CoppeliaSim. It isolates checkpoint loading + Dreamer +
CEM from the socket layer.

```bash
python -m app.vjepa_2_1_dreamer_ac.deploy --fname configs/inference/deploy_vjepa_2_1.yaml --debugmode True
```

**✓ check:** prints `[CLIENT] step=0`, a `prev_goal.shape`, and an action vector,
then exits cleanly. If this fails, the problem is the model or checkpoints — not the simulator.

## 15. Build the simulator environment

Separate env: Python 3.10 with `pyrep` + `rlbench` + CoppeliaSim and **no torch**.
It talks to `deploy.py` only over `localhost:9001`, so the two dependency stacks
never interact.

```bash
bash server/install_sim_env.sh
```

**✓ check:** ends with `sim env OK` and prints two values. **Paste them into the
top of `server/run_rollout.py`** — `PY_SIM` and `COPPELIASIM_ROOT` are the only
real placeholders left in the repo.

The script does system libs (needs `sudo`), the `libffi7` backport, CoppeliaSim
4.1, the three env vars, then PyRep and RLBench from source. If you have no sudo
on the lab machine, run `bash server/install_sim_env.sh --skip-apt` and send the
package list to your admin.

> **The two things that go wrong here:**
> - **`libffi7`.** CoppeliaSim 4.1 links `libffi.so.7`; Ubuntu 22.04+ ships
>   libffi8. Without the backported `.deb` CoppeliaSim exits immediately with an
>   error that never mentions libffi. The reference Colab run needed exactly this.
> - **The three env vars.** `COPPELIASIM_ROOT`, `LD_LIBRARY_PATH`, and
>   `QT_QPA_PLATFORM_PLUGIN_PATH` are read by PyRep at **build** time as well as
>   run time. If PyRep is pip-installed before they are exported, it compiles
>   against nothing and fails at import. The script appends them to `~/.bashrc`.

Confirm CoppeliaSim itself starts headless before trusting the whole stack:

```bash
Xvfb :99 -screen 0 1400x900x24 -ac -noreset > /dev/null 2>&1 &
```

```bash
DISPLAY=:99 $COPPELIASIM_ROOT/coppeliaSim.sh -h -q & sleep 15; pkill -f coppeliaSim && echo "CoppeliaSim headless OK"
```

**✓ check:** prints `CoppeliaSim headless OK`. A crash here is a graphics/library
problem, not an RLBench one — much easier to diagnose now than mid-rollout.

## 16. Check Xvfb

CoppeliaSim needs a display even headless.

```bash
which Xvfb || echo "MISSING - install xvfb"
```

```bash
Xvfb :99 -screen 0 1400x900x24 -ac -noreset > /dev/null 2>&1 &
```

```bash
sleep 3 && DISPLAY=:99 xdpyinfo | head -3
```

**✓ check:** `xdpyinfo` prints display info rather than "unable to open display".
`run_rollout.py` starts Xvfb itself, but verifying now saves debugging later.

## 17. Run rollouts

Switch back to the training env first.

```bash
conda activate djepa && python server/run_rollout.py --episodes 10 --task push_button --fresh
```

**✓ checks:**
- `xvfb: started` / `already running`
- `simulator starting...` then the policy runs — if it times out waiting for
  `:9001`, read `rollouts/server_ep0.log`
- each episode ends `-> SUCCESS` or `-> FAIL` with a nonzero frame count
- **zero frames on every episode** means `--save_image_dir` isn't writing — check the server log
- final `SUCCESS RATE: n/10`

Expect this to be slow: 200 samples × 50 CEM steps = 10,000 predictor forwards
per environment step.

## 18. Make the video

```bash
python server/make_video.py --frames rollouts/ep0 --reference data/val/push_button/sawyer/variation0_0000.hdf5
```

**✓ check:** writes `rollouts/ep0.gif`. Left panel is the sawyer reference demo,
right is the Franka execution. Add `--mp4` for video.

> Naming: the dataset directory is `franka/` while `server.py --robot` expects
> `panda`. Same arm — Franka Emika Panda — two naming conventions. Do not
> "fix" either one.

---

## Deviations to report

1. **4 GPUs instead of 8**, compensated by gradient accumulation — global batch
   is 128 (stage 1) and 16 (stage 2), matching the paper.
2. **Stage 0 not trained**; started from released V-JEPA 2.1-AC weights.
3. **`float16` instead of `bfloat16`** — only if step 2 showed V100s.

Everything else — model, resolution, epochs, LR schedule, losses, MPC settings —
is upstream.

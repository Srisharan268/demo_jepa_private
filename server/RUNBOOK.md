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

```bash
gh repo create your-name/demo-jepa-lab --private --source=. --remote=lab --push
```

**✓ check:** the last command prints your repo URL, and the repo shows a
`server/` folder with 11 files.

> GitHub's Fork button cannot make a public repo private — that is why this
> creates a fresh private repo rather than forking. `origin` still points at
> upstream, so `git fetch origin` continues to work for checking drift.

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

```bash
git clone https://github.com/your-name/demo-jepa-lab.git ~/Demo-JEPA
```

**✓ check:** `ls ~/Demo-JEPA/server/` shows 11 files. **From here on, every
command assumes you are in `~/Demo-JEPA` on the server.**

## 4. Build the conda environment

```bash
conda create -n djepa python=3.12 -y && conda activate djepa
```

```bash
cd ~/Demo-JEPA && pip install . && pip install diffusers==0.11.1
```

**✓ check:** both finish without a resolver error.

## 5. Fix the diffusers import

Open `$CONDA_PREFIX/lib/python3.12/site-packages/diffusers/dynamic_modules_utils.py`
and delete the line importing `cached_download` from `huggingface_hub`.

```bash
python -c "import torch, diffusers, timm, einops, h5py, decord; print('imports OK', torch.__version__, torch.cuda.is_available())"
```

**✓ check:** prints `imports OK <version> True`. `False` means torch can't see the GPUs — stop.

## 6. Get the Stage 0 checkpoint

Download it **on the server**, not via your laptop — it is a public file and the
lab's bandwidth will beat your home upload. Then strip the optimizer state
(~11GB → ~2–3GB):

```bash
python -c "import torch,sys; ck=torch.load(sys.argv[1],map_location='cpu',mmap=True); torch.save({k:ck[k] for k in ('target_encoder','predictor') if k in ck}, sys.argv[2]); print('keys:', sorted(ck.keys()))" /path/to/vjepa2_ac.pt ~/vjepa2_ac_repacked.pt
```

**✓ check:** printed keys include `target_encoder` and `predictor`; the repacked
file is ~2–3GB. If either key is missing, the config's `target_encoder_key`
won't match and stage 1 will fail at load.

## 7. Get the dataset in place

Your RLBench pairs must be on the server in aloha hdf5 layout, split into a
**training set** and a **held-out set** — Section 2 needs the held-out set for
both the retrieval eval and the one-shot reference demo.

The datasets are NOT in the git repo (`.gitignore` excludes `*.hdf5`), so pick
whichever source applies:

**(a) Already on Google Drive** — pull straight to the server, skipping your laptop:

```bash
pip install gdown && gdown --folder "<drive-folder-url>" -O ~/data
```

**(b) On a local disk** — one rsync from your laptop. It resumes if interrupted:

```bash
rsync -avP /path/on/laptop/ you@server:~/data/
```

**(c) Not collected yet** — generate on the server. This is synthetic data and
`--seed_master` makes it reproducible, so generating here is equivalent to
copying. **But it needs the simulator environment first**, so do Section 2
steps 15–16 before this, then use the collection command from the main
[README.md](../README.md) with `--headless`.

```bash
ls ~/data && du -sh ~/data
```

**✓ check:** per-task directories containing `panda/` and `sawyer/` subdirs with
`.hdf5` files, and a separate held-out directory.

## 8. Write the training configs

Edit the paths at the top of `server/prepare_configs.py` — `DATASET`,
`STAGE0_CKPT`, `OUT_STAGE1`, `OUT_STAGE2`. Then:

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
bash server/run_eval_stage1.sh /path/to/held_out_data
```

**✓ check:** prints `acc@1` well above chance. The script prints chance for you
(`1/batch_size` = 0.0625 at batch 16). At or near chance means the Dreamer
Predictor didn't learn — investigate before spending time on rollouts.

## 12. Build the deploy checkpoint

```bash
python server/make_deploy_ckpt.py /path/to/exp/stage2/latest.pt ~/stage2_deploy.pt
```

**✓ check:** prints the written size (~5–6GB). Errors here mean stage 2's
checkpoint lacks `target_encoder` or `predictor`.

## 13. Write the deploy config

Edit paths at the top of `server/prepare_deploy_config.py` — `DEPLOY_CKPT`,
`STAGE1_CKPT`, `REFERENCE_H5` (one held-out **sawyer** episode: the one-shot
prompt), `OUT_FOLDER`. Then:

```bash
python server/prepare_deploy_config.py
```

**✓ check:** all three paths `[ok]`, and mpc reads
`samples: 200, cem_steps: 50, topk: 10` — the paper's values.

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
Follow the RLBench install guide, including its headless-rendering section.

```bash
conda create -n rlbench python=3.10 -y && conda activate rlbench
```

```bash
/opt/conda/envs/rlbench/bin/python -c "import pyrep, rlbench; print('sim env OK')"
```

**✓ check:** prints `sim env OK`. Then set `PY_SIM` and `COPPELIASIM_ROOT` at the
top of `server/run_rollout.py` to match.

## 16. Check Xvfb

CoppeliaSim needs a display even headless.

```bash
which Xvfb && Xvfb :99 -screen 0 1400x900x24 -ac & sleep 3 && DISPLAY=:99 xdpyinfo | head -3
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
python server/make_video.py --frames rollouts/ep0 --reference /path/to/held_out/push_button/sawyer/episode0.hdf5
```

**✓ check:** writes `rollouts/ep0.gif`. Left panel is the sawyer reference demo,
right is the panda execution. Add `--mp4` for video.

---

## Deviations to report

1. **4 GPUs instead of 8**, compensated by gradient accumulation — global batch
   is 128 (stage 1) and 16 (stage 2), matching the paper.
2. **Stage 0 not trained**; started from released V-JEPA 2.1-AC weights.
3. **`float16` instead of `bfloat16`** — only if step 1 showed V100s.

Everything else — model, resolution, epochs, LR schedule, losses, MPC settings —
is upstream.

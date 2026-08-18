# Smoke run — plain commands, in order

Single 16GB GPU (RTX 4080 Super). Proves the pipeline runs end to end.
It does **not** train anything useful: 20 optimizer steps at batch 1.

Run every command from the repo root: `cd ~/Demo_JEPA/Demo-JEPA`

Each step says what you should see. If a step's check fails, stop there.

---

## 1. Conda env with the right torch

Your driver is 570.211.01 → CUDA **12.8**. Confirm with the header of plain `nvidia-smi`:

```bash
nvidia-smi | head -3
```

Create the env (do not use conda base):

```bash
conda create -n djepa python=3.12 -y
```

```bash
conda activate djepa
```

Install torch matching the driver **before** anything else:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Check:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print `True` and `NVIDIA GeForce RTX 4080 SUPER`. If `False`, the cu version is wrong — use whatever `nvidia-smi` reported (12.6 → `cu126`).

---

## 2. Install the repo

```bash
pip install .
```

**Check:**

```bash
python -c "import timm, einops, h5py, decord, transformers, yaml; print('deps OK')"
```

```bash
python -c "import app.vjepa_2_1_dreamer_predictor.train, app.vjepa_2_1_dreamer_ac.train; print('repo modules OK')"
```

Both must print. Skip `diffusers` — it is only used by the imitation baseline.

---

## 3. Dataset

```bash
cd data && tar -xzf rlbench_data.tar.gz && cd ..
```

```bash
python server/split_dataset.py
```

**Check:** prints `push_button   train 12   held-out 6`, and `data/train` + `data/val` now exist.

---

## 4. Stage 0 checkpoint

You already have a partial download. Resume it (no `-O`; `-c` resumes):

```bash
cd ~ && wget -c --progress=dot:giga https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt && cd -
```

**Check — must be ~11GB, not 3.4GB:**

```bash
ls -lh ~/vjepa2-ac-vitg.pt
```

If it is short, rerun the wget line. If it refuses to grow, `rm` it and start over.

Repack (drops optimizer state, aliases `target_encoder`, fixes key names):

```bash
python server/repack_stage0.py ~/vjepa2-ac-vitg.pt ~/vjepa2_ac_repacked.pt --bf16
```

**Check — the most important line in this whole runbook:**

```
top-level keys: ['encoder', 'predictor', ...]
sample key    : module.<something>
```

`sample key` **must** start with `module.`. Without it the weights load "successfully" while binding nothing, and you get a loss that never moves instead of an error. The script exits if it is wrong.

---

## 5. 16GB patches

These make it fit in 16GB. **Smoke test only** — they include a bf16 trainable
predictor, which is a real deviation.

```bash
git checkout -b test-16gb
```

```bash
python server/patch_16gb.py
```

**Check:** `stage2 bf16 casts=3 (want 3)`, `mask64=True`, `scaler bypassed=True`.

---

## 6. Configs

```bash
python server/prepare_configs.py --gpus 1 --smoke
```

**Check:** `batch_size=1 x 1 GPUs x accum 2`, dataset `[ok ]`, pretrain_checkpoint `[ok ]`,
and `12 batches/rank/epoch` (must not be 0 — 0 means it would hang).

If `pretrain_checkpoint` shows `[MISSING]`, step 4 did not finish.

---

## 7. Stage 1

```bash
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

```bash
python -m app.main --fname configs/train/vjepa_2_1_dreamer_predictor.yaml --devices cuda:0 --debugmode True 2>&1 | tee stage1_smoke.log
```

**Check:**
- `loaded pretrained` with keys matched, no `size mismatch`
- 20 lines of `loss:` — value is meaningless, it just has to be a number and not `nan`
- **record the `[iter: ... ms]` and `[mem: ...]` numbers** — first real measurements
- `exp/stage1/latest.pt` exists afterwards

```bash
ls -lh exp/stage1/
```

If it OOMs: patches from step 5 did not apply. Rerun `python server/patch_16gb.py --check`.

---

## 8. Verify stage 1's checkpoint

```bash
python -c "import torch; ck=torch.load('exp/stage1/latest.pt',map_location='cpu',mmap=True); print(sorted(ck.keys())); print('epoch',ck.get('epoch'),'loss',ck.get('loss'))"
```

**Check:** keys include `encoder` and `dreamer_predictor`.

---

## 9. Stage 2

```bash
python -m app.main --fname configs/train/vjepa_2_1_dreamer_ac.yaml --devices cuda:0 --debugmode True 2>&1 | tee stage2_smoke.log
```

**Check:**
- `loaded pretrained predictor` and `loaded pretrained dreamer_predictor`, keys matched
- loss lines show a `[a, b]` pair (teacher-forcing, autoregressive)
- `exp/stage2/latest.pt` exists

NCCL teardown can hang for a few minutes after `avg. loss`. That is normal — results are already saved.

---

## 10. Deploy checkpoint

```bash
python server/make_deploy_ckpt.py exp/stage2/latest.pt exp/stage2_deploy.pt
```

**Check:** prints a size in GB.

---

## 11. CEM without the simulator

This isolates model + CEM from the socket layer. Do it before touching CoppeliaSim.

```bash
python server/prepare_deploy_config.py
```

**Check:** all three paths `[ok]`, reference resolves under `data/val/push_button/sawyer/`.

```bash
python -m app.vjepa_2_1_dreamer_ac.deploy --fname configs/inference/deploy_vjepa_2_1.yaml --debugmode True
```

**Check:** prints `[CLIENT] step=0`, a `prev_goal.shape`, an action vector, exits cleanly.

**If everything up to here passes, the model half of the pipeline works.**

---

## 12. Simulator (only if RLBench is installed)

Restore the prebuilt env (558MB tarball on your Drive):

```bash
bash server/use_prebuilt_sim.sh ~/rlbench_env.tar.gz
```

Or build from source: `bash server/install_sim_env.sh`

**Check:** prints `sim env OK` and two paths. Paste them into the top of
`server/run_rollout.py` (`PY_SIM`, `COPPELIASIM_ROOT`).

```bash
Xvfb :99 -screen 0 1400x900x24 -ac -noreset > /dev/null 2>&1 &
```

```bash
DISPLAY=:99 xdpyinfo | head -3
```

**Check:** prints display info, not "unable to open display".

---

## 13. One rollout

```bash
python server/run_rollout.py --episodes 1 --task push_button --fresh
```

**Check:** ends `-> SUCCESS` or `-> FAIL` with a **nonzero frame count**. FAIL is
expected — the model has had 20 training steps. Zero frames means
`--save_image_dir` is not writing; read `rollouts/server_ep0.log`.

```bash
ls rollouts/ep0 | head
```

---

## 14. Video

```bash
python server/make_video.py --frames rollouts/ep0 --reference data/val/push_button/sawyer/variation0_0000.hdf5
```

**Check:** writes `rollouts/ep0.gif`. Left = sawyer reference demo, right = franka execution.

**A gif at the end means the whole pipeline works.** The motion will look random. That is a pass.

---

## Afterwards

Report back:
- `[iter: ... ms]` and `[mem: ...]` from step 7
- anything that failed, and at which step

Then discard the test branch so the patches never reach the real run:

```bash
git checkout server-4gpu
```

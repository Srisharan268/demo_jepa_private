# Server run — Demo-JEPA stages 1 & 2 on 4×32GB

Reproduces the paper's Dreamer pipeline on 4 GPUs instead of 8. Stage 0 is not
trained; it loads Meta's released V-JEPA 2.1-AC checkpoint. The imitation
experiment is a Diffusion Policy baseline and is out of scope, so the
`diffusion_policy` submodule does not need initialising.

## Deviations from upstream

Exactly **one** code change, plus config values:

| Change | Where | Why |
|---|---|---|
| gradient accumulation | `app/vjepa_2_1_dreamer_predictor/train.py` | Stage 1 cannot hold batch 32/GPU in 32 GB; accumulation restores global batch 128 |
| `batch_size` 16→8, `accum_steps: 4` | stage 1 config | 8 × 4 GPUs × 4 = 128 = paper |
| `batch_size` 2→4 | stage 2 config | 4 × 4 GPUs = 16 = paper. No code change needed |
| `camera_views`, `data_type`, paths | both configs | environment; `camera_front` does not exist in RLBench output |

`accum_steps` defaults to `1`, so the patched file is byte-equivalent in
behaviour to upstream unless a config sets it.

**Unchanged:** model, `crop_size`, `dataset_fpcs`, `epochs`, `ipe`, `lr`,
`warmup`, `anneal`, weight decay, losses, `dtype: bfloat16`.

## Order of operations

1. **Check the GPUs.** If they report as V100 (`sm_70`), bf16 is unsupported and
   every config's `dtype: bfloat16` must become `float16` — a real deviation, so
   confirm before running.

   ```bash
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
   ```

2. **Extract and split the dataset.** The archive ships in the repo.

   ```bash
   cd data && tar -xzf rlbench_data.tar.gz && cd .. && python server/split_dataset.py
   ```

3. **Write the configs.** Paths are repo-relative — nothing to edit unless the
   Stage 0 checkpoint is somewhere other than `~/vjepa2_ac_repacked.pt`. Asserts
   upstream values first, so it fails loudly if the repo drifted.

   ```bash
   python server/prepare_configs.py
   ```

4. **Measure before committing to the long run.** Stage 1 is the tight one;
   estimated ~24–27 GB at `batch_size: 8`. Run a few steps and read the
   `[mem: ...]` field the training loop already logs (MB, from
   `torch.cuda.max_memory_allocated`). If it OOMs, drop to `batch_size: 4` and
   raise `accum_steps` to `8` — global batch stays 128.

5. **Run stage 1**, then stage 2, each in its own tmux session.

   ```bash
   tmux new -s djepa
   ```

   ```bash
   bash server/run_stage1.sh 2>&1 | tee stage1.log
   ```

   Detach with `ctrl-b d`, reattach with `tmux attach -t djepa`.

6. **Run stage 2** once stage 1 has written `latest.pt`. The script checks for it.

   ```bash
   bash server/run_stage2.sh 2>&1 | tee stage2.log
   ```

## Evaluation

### Stage 1 — cross-embodiment retrieval

Uses the paper's own `app/vjepa_2_1_dreamer_predictor/retrieval_eval.py`. Reports
acc@1 / acc@k on **held-out** episodes; chance is `1/batch_size`.

```bash
bash server/run_eval_stage1.sh data/val
```

### Stage 2 — closed-loop success rate

This is the headline number: the fraction of episodes the policy completes from
a single reference demonstration of a *different* embodiment.

1. Extract a deploy checkpoint from stage 2's output:

   ```bash
   python server/make_deploy_ckpt.py exp/stage2/latest.pt exp/stage2_deploy.pt
   ```

2. Write the deploy config (paths are repo-relative, reference auto-selected):

   ```bash
   python server/prepare_deploy_config.py
   ```

3. Validate the CEM path **without** the simulator first — this isolates
   checkpoint loading + Dreamer + CEM from the socket layer and saves hours when
   something is wrong:

   ```bash
   python -m app.vjepa_2_1_dreamer_ac.deploy --fname configs/inference/deploy_vjepa_2_1.yaml --debugmode True
   ```

4. Run the real rollouts:

   ```bash
   python server/run_rollout.py --episodes 10 --task push_button --fresh
   ```

## Visual rollout

`deploy.py` saves nothing itself — frames come from the simulator side via
`server.py --save_image_dir`, which `run_rollout.py` sets per episode. Then:

```bash
python server/make_video.py --frames rollouts/ep0 --reference data/val/push_button/sawyer/variation0_0000.hdf5
```

Produces a side-by-side: reference demo (source embodiment) on the left, policy
execution (target embodiment) on the right — the comparison the paper is about.
Pass `--mp4` for video instead of GIF.

### Requirements for rollouts

- **A second conda env** for the simulator: `pyrep` + `rlbench` + CoppeliaSim,
  Python 3.10, **no torch**. `deploy.py` runs in the training env. They
  communicate only over `localhost:9001`, so the stacks never interact. Set
  `PY_SIM` and `COPPELIASIM_ROOT` at the top of `run_rollout.py`.
- **Xvfb** — CoppeliaSim will not start without a display, even headless.
  `run_rollout.py` starts one on `:99` automatically.
- **Time.** MPC settings are the paper's: `samples: 200`, `cem_steps: 50` = 10,000
  predictor forwards *per environment step*. `prepare_deploy_config.py` asserts
  these rather than trimming them. Reducing them is a reported-result change, not
  a free speedup.

## Notes

- **Run each stage as one continuous job.** `load_checkpoint` always reports
  `start_epoch = 0`, so resuming restarts the LR schedule from scratch. Chunking
  a run turns one cosine cycle into several — a real deviation.
- **Do not launch from a notebook.** `app/main.py` uses `mp.set_start_method("spawn")`,
  and spawn re-imports `__main__`, which is not a file in a Jupyter kernel. The
  single-GPU `--debugmode True` path avoids multiprocessing entirely and is fine
  in a notebook, but it is not what you want on 4 GPUs.
- **`scripts/*.sh` are broken upstream** — three of them point at
  `configs/train/vjepa_2_1/*.yaml`, but the files live at `configs/train/*.yaml`.
  The scripts here use the correct paths, so those are left untouched.
- **Host RAM**: see `server/optional_mask_size.py` if stage 2 dies at startup.
- Checkpoints land in `folder/latest.pt` every epoch (`CHECKPOINT_FREQ = 1`,
  `ipe: 300`), plus `e{N}.pt` every `save_every_freq: 25` epochs.

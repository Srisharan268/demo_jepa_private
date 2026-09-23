# Lab runbook — 1× 32 GB GPU, Docker

Ordered so that you have a **showable result within the first day**, and so that
every expensive step sits behind a cheap test that can stop it. Each step says
what "good" looks like and what to do if it isn't.

Branch: **`lab`**. Everything below runs from the repo root on the lab machine.

> **Nothing in this branch has been executed yet.** It was written on a machine
> with no torch, numpy, or Docker — syntax-checked, not run. Steps 3–4 exist to
> catch that before it costs you anything. Treat a failure there as expected,
> not alarming.

---

## 0. Before you leave your PC

```bash
git push lab refs/heads/lab:refs/heads/lab
```

(The branch and the remote are both named `lab`, so the short form is ambiguous.)

Files to copy to the lab machine (from the Windows repo folder):

| file | size | what it is |
|---|---|---|
| `data.tar` | 4.6 GB | 362 train + 40 val episode pairs, `franka/`+`sawyer/` layout |
| `vjepa2_ac_repacked.pt` | 2.5 GB | stage 0: encoder + Meta-pretrained AC predictor |
| `stage1_slim.pt` | 164 MB | existing stage 1 dreamer (12 epochs) |

**Do not copy the repo folder itself** — clone it (step 1). Older files in the
Windows working tree have CRLF line endings, and every shell script fails on
Linux with `$'\r': command not found`. The copies in git are clean.

## 1. Clone

```bash
git clone -b lab https://github.com/Srisharan268/demo_jepa_private.git Demo-JEPA
cd Demo-JEPA
```

Private repo, so it prompts: username, then a **fine-grained, read-only, repo-scoped
personal access token** as the password. Don't paste the token into the URL — on a
shared machine it would be stored in `.git/config`.

## 2. Data and checkpoints

```bash
tar -xf ~/data.tar                      # -> data/train, data/val
mv ~/vjepa2_ac_repacked.pt ~/stage1_slim.pt .
ls data/train/push_button data/val/push_button    # expect: franka  sawyer
```

## 3. Build the image  (~20–40 min, mostly downloads)

```bash
docker build -f docker/Dockerfile -t demojepa .
```

- **Base tag not found** → any `nvidia/cuda` 12.8+ `ubuntu22.04` runtime tag works:
  `docker build --build-arg BASE=<tag> -f docker/Dockerfile -t demojepa .`
- **Fails at CoppeliaSim download** → the URL in `server/install_sim_env.sh` has
  moved (they change between releases). Find the 4.1.0 Ubuntu 20.04 build on
  coppeliarobotics.com and pass `COPPELIA_URL=...`, or restore the prebuilt
  `rlbench_env.tar.gz` with `server/use_prebuilt_sim.sh` if you still have it.
- **Permission denied on `docker`** → you're not in the `docker` group; ask the
  admin. Without Docker at all, see "No Docker" at the end.

## 4. Smoke-test the container  (5 min — do not skip)

```bash
bash docker/run.sh nvidia-smi
bash docker/run.sh python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); x=torch.randn(64,64,device='cuda'); print('matmul ok', float((x@x).sum()))"
bash docker/run.sh bash -c 'Xvfb :99 -screen 0 1400x900x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 & sleep 3; export DISPLAY=:99 LD_LIBRARY_PATH=$COPPELIASIM_ROOT QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT; $PY_SIM -c "import pyrep, rlbench" && echo SIM_OK'
```

Good: GPU listed; torch ≥ 2.7 with your card's name; `matmul ok`; `SIM_OK`.

- `no kernel image is available` → torch too old for the card (Blackwell needs cu128).
- Permission errors anywhere → retry with `RUN_AS_ROOT=1 bash docker/run.sh ...`.
  Outputs will then be root-owned; tell me and I'll fix the non-root path.

## 5. Check the data, then configure

```bash
bash docker/run.sh python -c "
import h5py, glob
v = sorted(glob.glob('data/val/push_button/franka/*.hdf5'))
a = dict(h5py.File(v[0]).attrs)
print(len(v), 'val episodes;', {k: a.get(k) for k in ('dt', 'episode_seed_used', 'variation', 'robot', 'source_robot')})
for s in ('franka', 'sawyer'):
    print('train', s, len(glob.glob(f'data/train/push_button/{s}/*.hdf5')))
"
```

Must be true: `dt: 0.05` (20 Hz), `episode_seed_used` present (step 9 needs it),
~40 val and ~362 train per robot. Tell me if any differ.

**Note `source_robot`.** If it is `sawyer` rather than the `franka` file's own
`robot`, the sawyer side holds the source demo, and step 9 needs
`--robot-subdir sawyer`. (This dataset has `franka/` directories, and a logged
collection command used `--source_robot sawyer`; the 4080's copy used `panda`.)

```bash
bash docker/run.sh python server/prepare_configs.py --gpus 1 --s2-batch 8
```

`--gpus 1` matters: the default is 4, which silently sizes accumulation and
batch for four cards. Batch 8 is **measured** on a 32 GB card for both stages
(16 OOMs). It should report stage 2 `data_fps 20` — read from the data, giving the
paper's 5 fps. **Every run of this script rewrites both stages' configs**, so run
it again immediately before each training step below.

## 6. Measure maxnorm  (2 min)

```bash
bash docker/run.sh python -c "
import h5py, glob, numpy as np, yaml
from math import ceil
d = yaml.safe_load(open('configs/train/vjepa_2_1_dreamer_ac.yaml'))['data']
k = ceil(d['data_fps'] / d['fps'])
a = np.concatenate([np.abs(np.diff(np.array(h5py.File(f)['observations/qpos'])[::k, :3], axis=0))
                    for f in glob.glob('data/val/push_button/franka/*.hdf5')])
print('stride', k, 'raw frames | per-axis |dxyz|  p95', np.percentile(a, 95, 0).round(4), ' p99', np.percentile(a, 99, 0).round(4))
"
```

Set `maxnorm` ≈ the **largest p99**. The template ships 0.05, estimated from one
episode. Pass what you measure as `--maxnorm` to `prepare_deploy_config.py` in
steps 7 and 11 — don't edit the yaml, it's read from git and the edit would
silently do nothing.

## 7. GATE A — does stage 1 actually use the demonstration?  (10 min)

The most important untested question. If the dreamer ignores the sawyer demo,
its goal is a function of the franka's own view, stage 2 can match it without
ever using actions, and **no amount of stage 2 training helps**.

```bash
bash docker/run.sh python server/prepare_deploy_config.py --task push_button --deploy-ckpt vjepa2_ac_repacked.pt --stage1-ckpt stage1_slim.pt
bash docker/run.sh python server/dreamer_test.py 2>&1 | tail -30
```

Read the two verdicts at the bottom:

- **`REFERENCE USE` ratio ≥ 0.1** → the dreamer conditions on the demo. Continue.
- **Below 0.1** → it largely ignores the demo. **Stop and send me the output.** A
  12-epoch dreamer may just be undertrained, but that needs deciding before any
  multi-day run.
- **`GOAL QUALITY` never beats identity** → its goals are worse than "predict no
  change". Also send me the output.

## 8. GATE B — does the pretrained AC predictor respond to actions?  (5 min)

The AC predictor starts Meta-pretrained. This measures whether it responds to
**our** actions before stage 2 touches it — the baseline stage 2 must not destroy.

```bash
bash docker/run.sh python server/val_loss.py --batches 8 2>&1 | tail -22
```

Look at `position 0` in the sensitivity table (the one-frame regime the planner
uses):

- **Clearly above 0** → it can condition on our actions. Stage 2's job is to keep that.
- **~0 even here** → a domain or action-convention gap between Meta's training
  data and RLBench. Stage 2 must build conditioning from scratch — a harder
  problem. Send me the output.

## 9. Recover the evaluation scenes  (~15 min)

Rebuilds the deleted `rng_state.pkl` for each held-out episode from the seed
stamped in its HDF5, **verified** by checking the regenerated demo ends where the
stored one did. Without it, every rollout runs in a random scene where the
button isn't where the demo put it.

```bash
bash docker/run.sh bash -c 'Xvfb :99 -screen 0 1400x900x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 & sleep 3; export DISPLAY=:99 LD_LIBRARY_PATH=$COPPELIASIM_ROOT QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT; $PY_SIM server/recover_rng.py --task push_button --limit 1'
```

If step 5 showed `source_robot: sawyer`, add `--robot-subdir sawyer`. (Without
it, every episode prints `SKIP -- ... not the source`: harmless, just rerun.)

If that prints `OK (self at ... margin ...x)`, rerun without `--limit 1`. On the
4080 this verified 6/6 at margins of 111× to infinite. Restoring the scene for
either robot is correct — both robots in a pair were recorded in the same scene.

## 10. Stage 2 PILOT — the go/no-go  (~1 h)

Trains stage 2 against the existing `stage1_slim.pt`, at the corrected 5 fps,
with the new action-response metric. **This is the decision point for committing
days of compute.**

```bash
bash docker/run.sh python server/prepare_configs.py --gpus 1 --s2-batch 8 --epochs 20 --ipe 100 --warmup 1 --anneal 1 --stage1-ckpt stage1_slim.pt
tmux new -s pilot            # ON THE HOST, so an SSH drop doesn't kill it
bash docker/run.sh bash -c 'bash server/run_stage2.sh 2>&1 | tee stage2_pilot.log'
```

Warmup and anneal are **absolute epochs**, not fractions — scale them with
`--epochs` (≈5% each), or the schedule is silently wrong.

Watch from another host terminal:

```bash
grep "action response" stage2_pilot.log | tail -5
```

`pos0` is in units of *one real frame of latent change*. The first value is the
pretrained predictor — your baseline.

- **Holds at or above ~half its first value** → **GO.**
- **Decays toward 0 in the first few epochs** → stage 2 is erasing the pretrained
  action conditioning — the exact failure of the last checkpoint. **Stop
  (Ctrl-C) and send me the log.** Don't start step 12.
- **OOM at iteration 0** → rerun step 10 with `--s2-batch 4`. The metric runs at
  iteration 0 precisely so memory problems surface immediately.

When it finishes:

```bash
bash docker/run.sh python server/check_stage2.py stage2_pilot.log
```

The `exp/stage1/latest.pt exists` line will FAIL — expected, the pilot used the
slim checkpoint. Everything else should pass.

## 11. First rollouts — your showable result  (1–3 h)

```bash
bash docker/run.sh python server/make_deploy_ckpt.py exp/stage2/latest.pt exp/stage2_deploy.pt
bash docker/run.sh python server/prepare_deploy_config.py --task push_button --stage1-ckpt stage1_slim.pt --max-steps 60 --maxnorm <from step 6>
bash docker/run.sh python server/run_rollout.py --task push_button --episodes 5 --out rollouts_pilot
bash docker/run.sh python server/make_gifs.py --folder rollouts_pilot/
```

`--stage1-ckpt` must be the **same** dreamer the pilot trained against.

What to read, in order of how informative it is:

1. `rollouts_pilot/deploy_ep0.log` — do actions still pin to `±maxnorm` every
   step? Varying magnitudes and headings that change as the demo curves mean the
   planner has something to optimise.
2. **Heading.** The push_button demo descends ~0.7 m. A policy moving *down*
   toward the button, even without pressing it, is the result to show.
3. `latent l1 dist` in the same log. If it's **always below** `l1_threshold`,
   the subgoal never gates and the demo is consumed every step — calibrate it
   (below) and pass `--l1-threshold`.
4. Success rate — the least informative number at 5 episodes.

Calibrating `l1_threshold` (only if 3 says so):

```bash
bash docker/run.sh python -c "
import yaml
from app.vjepa_2_1_dreamer_ac.deploy import build_world_model
from app.vjepa_2_1_dreamer_ac.cem_utils import calibrate_threshold_from_sequence
p = yaml.safe_load(open('configs/inference/deploy_vjepa_2_1.yaml'))
wm, _, _ = build_world_model(p)
d = p['deploy']
thr, stats, _, _ = calibrate_threshold_from_sequence(wm, d['reference_h5'], img_key=d['image_key'], data_fps=d['ref_data_fps'], target_fps=d['ref_target_fps'])
print('threshold', thr, stats)
"
```

Then, **before step 12**, move the pilot aside. Resume now works, so a new stage
2 run into the same folder would silently *continue the pilot* instead of
starting fresh:

```bash
mv exp/stage2 exp/stage2_pilot && mv exp/stage2_deploy.pt exp/stage2_pilot/
```

## 12. Full stage 1  (days)

Only after step 10 said GO. Stage 1 trains the dreamer from scratch and is where
the days go.

Measure speed first, rather than trust arithmetic. Start the run, read the
timing after ~15 minutes, then size it to your budget:

```bash
bash docker/run.sh python server/prepare_configs.py --gpus 1 --s2-batch 8 --ipe 100
tmux new -s stage1
bash docker/run.sh bash -c 'bash server/run_stage1.sh 2>&1 | tee -a stage1.log'
# ...15 minutes later, from another terminal:
grep "iter:" stage1.log | tail -3
```

With `T` = ms per logged iteration and `H` = hours available:

    epochs = floor(H × 3,600,000 / (100 × T))      warmup = anneal = max(1, round(0.05 × epochs))

Then stop it, discard the measurement run, and restart sized correctly:

```bash
rm -rf exp/stage1
bash docker/run.sh python server/prepare_configs.py --gpus 1 --s2-batch 8 --ipe 100 --epochs <E> --warmup <w> --anneal <w>
bash docker/run.sh bash -c 'bash server/run_stage1.sh 2>&1 | tee -a stage1.log'
```

**If it crashes or the machine reboots, just run the same command again.** It
resumes from `exp/stage1/latest.pt` — epoch, optimizer and LR schedule — and logs
`RESUMING from ... epoch N`. (Before this branch, a relaunch silently restarted
from epoch 0 with a fresh warmup.) Only `rm -rf exp/stage1` if you want a genuinely
new run.

Stage 1 logs val loss, `pred-std` (collapse monitor — must not trend to 0) and
`cos-sim` in `exp/stage1/log_r0.csv`. Rerun step 7's `dreamer_test.py` against
`exp/stage1/latest.pt` partway through to watch reference use improve.

## 13. Stage 2 on the new dreamer  (1–2 h)

```bash
bash docker/run.sh python server/prepare_configs.py --gpus 1 --s2-batch 8 --epochs 36 --ipe 100 --warmup 2 --anneal 2
tmux new -s stage2
bash docker/run.sh bash -c 'bash server/run_stage2.sh 2>&1 | tee stage2.log'
bash docker/run.sh python server/check_stage2.py stage2.log
```

No `--stage1-ckpt` → defaults to `exp/stage1/latest.pt`. Same action-response
watch as step 10.

## 14. Final rollouts

Step 11's commands without `--stage1-ckpt` (defaults to the new stage 1), and
drop `--episodes` to run every recovered scene.

---

## Reference

**Deviations from the paper, deliberately kept** (report them):
- 402 episode pairs, one task — ~1% of the paper's data.
- Global batch 128 via accumulation for stage 1; **8 for stage 2** (paper 16 — 16 OOMs on 32 GB).
- `maxnorm`, `l1_threshold`: the paper gives no values; ours are measured.
- Stage 0 repacked in bf16, so the trainable AC predictor's init is bf16-rounded.
  Small; fixable by re-running `repack_stage0.py` without `--bf16` on the 11 GB original.

**Fixed on this branch** (all were silently wrong before):
- Stage 2 sampled every raw frame of 20 Hz data (4× the paper's 5 fps).
- Deploy centre-cropped 25% of each frame's field of view; rollouts rendered at 256×256, not 640×480.
- Deploy's reference stride didn't match the training action stride.
- Resume restarted training from epoch 0 in both stages.
- Rollouts ran in random scenes, with one demo for every episode.

**No Docker.** The same recipe on the host: install Miniforge; then
`conda create -n djepa python=3.12 -y`, `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`,
`pip install -r requirements.txt`; then `bash server/install_sim_env.sh`
(`--skip-apt` without sudo). Export the `PY_SIM` and `COPPELIASIM_ROOT` it prints,
then drop the `bash docker/run.sh` prefix from every command above.

**Weights & Biases** is off by default. To enable:
`WANDB_MODE=online WANDB_API_KEY=<key> bash docker/run.sh ...`

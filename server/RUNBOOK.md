# RUNBOOK — rented-GPU measurement session

Branch `cloud-test`. Single runbook, supersedes the old `RUNBOOK.md` (written for
4 GPUs) and `SMOKE_RUNBOOK.md` (16GB smoke test, finished).

Full project context is in `HANDOFF.md`. Read §3, §5b, §6 and §11 of it before
this. This file is only the session procedure.

---

## 0. What this session is for

Get the **measurements** that let us predict full-scale results, on hardware
that is not shared and not memory-constrained. Nothing here is a result to
report — it is instrumentation for deciding how to spend real money.

| | |
|---|---|
| Hardware | vast.ai **RTX 5090, 32 GB, ~107 TFLOPS** |
| Budget | ₹1000 ≈ $12. **Target spend 4–5 h; keep the rest as reserve.** |
| Lab card (the real target) | RTX PRO 4500 Blackwell **32 GB**, same architecture |

### Why a 5090 rather than a 96 GB card

**Because it has exactly the lab card's 32 GB.** Whatever batch fits the 5090 is
what fits the lab card — you measure the real constraint instead of extrapolating
around it. Both are Blackwell, so the spec-ratio step in §4.5 stays honest.

The 96 GB RTX PRO 6000 is only ~11% faster on compute (119 vs 107 TFLOPS); its
advantage is memory, and memory is not what you are short of.

There is also a failure mode worth finding cheaply: §6 predicts batch 8 needs
**24–27 GB**, from meta-tensor analysis that has never been checked against
reality. If that is optimistic and batch 8 does NOT fit 32 GB, the entire lab
plan is wrong. Far better to learn that on a cheap rental than on the lab card.

**What you give up:** you cannot measure batch 32, so you cannot answer "would a
bigger rented card beat the lab card". Secondary — the lab card cannot run
batch 32 either.

32 GB means **no memory hacks anyway**: `patch_16gb.py` is deliberately deleted
from this branch. Everything runs full fp32, exactly as the lab card will.

---

## 0b. Instance requirements — check BEFORE renting, verify AFTER

### Choosing the instance

| Setting | Minimum | Why |
|---|---|---|
| GPU | **RTX 5090, 32 GB** | same VRAM as the lab card — the whole point |
| **CUDA driver** | **≥ 12.8** | 5090 is Blackwell `sm_120`. Older drivers have no kernels for it |
| **System RAM** | **≥ 32 GB** | `max_num_frames=512` builds a `torch.zeros(66048, 66048)` causal mask = **17.4 GB of HOST RAM** (`ac_predictor.py:156` slices before `.to(device)`), plus ~5 GB of dataloader workers |
| **Disk** | **≥ 80 GB** | 7 GB transfer + ~36 GB checkpoints + headroom. **Cannot be raised later** |
| Image | `vastai/pytorch`, tag with **CUDA ≥ 12.8 and torch ≥ 2.7** | anything older fails at the first kernel |

There is no Ubuntu selector — the OS comes from the image. The glibc constraint
in §9 of HANDOFF applies only to CoppeliaSim on the 4080, not here.

Check available tags:

```bash
curl -s "https://hub.docker.com/v2/repositories/vastai/pytorch/tags?page_size=100" | python -c "import json,sys; [print(t['name']) for t in json.load(sys.stdin)['results']]"
```

### Verify on the box — first command after SSH

```bash
python -c "
import torch, psutil, shutil
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('gpu  ', torch.cuda.get_device_name(0))
free_gb = torch.cuda.get_device_properties(0).total_memory/1e9
print(f'vram  {free_gb:.1f} GB   ram {psutil.virtual_memory().total/1e9:.1f} GB   disk {shutil.disk_usage(\".\").free/1e9:.1f} GB free')
x = torch.randn(64, 64, device='cuda'); print((x@x).sum().item(), '<- CUDA KERNELS OK')
assert free_gb > 30, 'wrong GPU'
"
```

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv    # MUST be empty
```

**The matmul is the test that matters.** `torch.cuda.is_available()` returns
`True` even when no kernel exists for the architecture — it would pass on a
too-old image and then fail at the first real op, after you have transferred
7 GB. Run the matmul before transferring anything.

If it prints `no kernel image is available for execution on the device`,
**destroy the instance and pick a newer tag.** Do not debug the install on paid
time.

If RAM is below 32 GB and you cannot get more, `python server/optional_mask_size.py`
drops `max_num_frames` 512 → 64. That is a change to upstream behaviour, so it
is a fallback, not a plan — prefer paying for the RAM.

---

## 1. GATE: you need more data before renting

**Do not rent until this is done.** Throughput measured on the current 18-episode
smoke set is invalid:

- 12 training pairs < batch 16 → **0 batches** → `prepare_configs.py` exits 1
  (correctly — `drop_last=True` would hang silently, HANDOFF §6)
- batch 8 gives **1 batch/epoch** → the loader hits `StopIteration` and rebuilds
  every single step. That inflates `data` time and pollutes `iter`.

**Minimum: ~320 paired episodes in `data/train`** (20 batches/epoch at batch 16,
the largest size in the §4.1 sweep). Ideal is the full 4-task x 800-pair set,
which is also the real run's dataset — collect that if time allows.

Collect on the **4080 box** — this is CPU/sim work, needs no GPU, costs nothing,
and runs while you do everything else:

```bash
cd ~/Demo-JEPA
python scripts/rlbench_tools/cli.py --help
```

**Time 10 episodes first.** This number has never been measured and it gates the
entire project (HANDOFF §11.7):

```bash
time python scripts/rlbench_tools/cli.py <args for 10 episodes of push_button>
```

Record: seconds/episode, for both `franka` and `sawyer`. Then launch the real
collection under `tmux`. Re-split when done:

```bash
python server/split_dataset.py
```

---

## 2. Pre-flight — compose these BEFORE the meter starts

Have every command written down. Do not debug on paid time.

**2.1 Pick an image with PyTorch + CUDA already installed.** Building an
environment costs more than the measurements are worth. Verify `torch.__version__`
and `torch.cuda.is_available()` in the first minute.

**2.2 Transfer the checkpoint and data.** Everything is on the laptop already;
nothing needs re-downloading. **~7 GB total** — start it early, it is the longest
single step.

| What | Size | Destination on the box |
|---|---|---|
| `vjepa2_ac_repacked.pt` | 2.64 GB | `Demo-JEPA/` (repo root) |
| `data/train` | 3.9 GB | `Demo-JEPA/data/train` |
| `data/val` | 447 MB | `Demo-JEPA/data/val` |

**Repo root, not the home directory.** `prepare_configs.py` looks for the
checkpoint at `<repo>/vjepa2_ac_repacked.pt` first (it falls back to `~`, but
repo-relative means the same config works on every machine).

vast.ai gives you `ssh -p PORT root@HOST`. **`scp` uses capital `-P`** for the
port, unlike `ssh`. Clone the repo on the box FIRST (§3) so `Demo-JEPA/` exists:

```bash
scp -P <PORT> "C:\Users\srish\vjepa2_ac_repacked.pt" root@<HOST>:Demo-JEPA/
scp -P <PORT> -r "C:\WSAIS Intern\Demo-JEPA\data" root@<HOST>:Demo-JEPA/
```

Tar the data first if the link is slow or flaky — one stream beats thousands of
small files and resumes as a single unit:

```bash
tar -czf data.tgz data/train data/val
scp -P <PORT> data.tgz root@<HOST>:Demo-JEPA/
# on the box:  tar -xzf data.tgz && rm data.tgz
```

**Verify on the box before spending anything:**

```bash
python -c "
import os, torch
p = 'vjepa2_ac_repacked.pt'; print(f'{os.path.getsize(p)/1e9:.2f} GB')
e = torch.load(p, map_location='cpu', weights_only=False, mmap=True)['encoder']
assert 'module.norms_block.3.weight' in e and 'module.norm.weight' not in e
print('checkpoint OK (section 5 rename present)')"
find data/train -name '*.hdf5' | wc -l    # must match the laptop's count
```

If that assert fails the checkpoint predates commit `939124c` — regenerate it
with `repack_stage0.py` rather than training against a degraded encoder.

**2.4 wandb — set this up first, it is your only live backup.**

Use **your own** account. (The 4080's `~/.netrc` holds someone else's
credentials — HANDOFF §5b.) On the rented box:

```bash
pip install wandb && wandb login          # paste your key from wandb.ai/authorize
unset WANDB_MODE                          # the run scripts default it to disabled
```

`train.py` already calls `wandb.log` every 10 iterations with `loss`, `lr`,
`wd`, `grad_norm_unclipped`, `grad_norm_clipped`, `mem`, `iter`, `gpu`, `data`.
**Four of those (`lr`, `wd`, both grad norms) are not written to the CSV at
all** — without wandb they exist only in terminal scrollback.

**Keep in mind during the session:**

- **`WANDB_MODE` defaults to `disabled` in the run scripts** (`${WANDB_MODE:-disabled}`).
  Export `WANDB_MODE=online` explicitly, or you will discover at teardown that
  nothing was recorded. `sweep_memory.py` sets `disabled` on purpose — sweep
  configs are noise, not experiments.
- **Check the startup line.** train.py now logs
  `wandb run: <name>-b8xa16-MMDD-HHMMSS (mode=online)`. If it says
  `mode=disabled`, stop and fix it before the run.
- **Run names are now unique** and carry batch/accum plus a timestamp. The
  config ships `name: test`, so without this a sweep produced N runs all called
  "test". The full config is logged too, so you can tell afterwards which
  settings produced a curve.
- **wandb is your only live backup.** The console log and CSV die with the
  instance. Still `tee` every run and `scp` the logs at teardown — belt and
  braces.
- **Offline fallback:** if login fails or the box has no outbound network, use
  `WANDB_MODE=offline`. It writes to `./wandb/` locally; sync later with
  `wandb sync wandb/offline-run-*`. Do not just fall back to `disabled`.
- **`std`, `cos`, `VAL`, `top1/top5`, `lr`, `wd`, `grad` reach wandb and the
  console but NOT the CSV.** Without wandb they exist only in scrollback.

---

## 3. Box setup

```bash
git clone -b cloud-test https://github.com/Srisharan268/demo_jepa_private.git Demo-JEPA
cd Demo-JEPA
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Do not reinstall torch** if the image already has a working CUDA build —
that is the single easiest way to burn 20 minutes of paid time and end up with
a CPU-only wheel. Install everything else:

```bash
pip install -r requirements.txt
```

If that tries to pull a different torch, install the rest without it:

```bash
grep -v '^torch' requirements.txt | pip install -r /dev/stdin
```

Then confirm the imports that have actually broken before (`decord` fails on
some platforms, `cv2` needs the headless build in containers):

```bash
python -c "import torch, decord, cv2, h5py, pandas, timm, wandb, yaml, einops; print('imports OK')"
```

`pip install opencv-python-headless` if `cv2` fails in a headless container.

```bash
python -m wandb login          # see §2.4
export WANDB_MODE=online
```

Sanity-check the repo itself before spending anything:

```bash
python server/check_metrics.py --offline
```

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

**That must be empty.** A contended GPU is what made every previous measurement
worthless (HANDOFF §5b). If anything is running, stop and get a clean box.

```bash
tar -xf data.tar && rm data.tar          # if you transferred it as a tarball
find data/train -name '*.hdf5' | wc -l   # must match the laptop's count
find data/val   -name '*.hdf5' | wc -l
python server/prepare_configs.py --gpus 1
```

**Do NOT run `split_dataset.py` here** if you already split on the laptop (§1).
The transferred `data/train` and `data/val` are already separated; running it
again would carve a SECOND held-out set out of train and move it to val.

Check the reported `episode pairs -> N/rank -> M batches/rank/epoch`. If `M` is
small, go back to §1.

Optional, if host RAM is tight on the instance: `python server/optional_mask_size.py`
(`max_num_frames` 512→64 builds a 17.4 GB **host** RAM causal mask otherwise —
HANDOFF §6).

---

## 4. Measurements

Order matters. **Always run `set_batch.py` *after* `prepare_configs.py`** —
`prepare_configs.py` reads its baseline from `git HEAD` and rewrites the file,
so running it again undoes your batch choice.

```bash
python server/set_batch.py --batch 8 --measure
```

`--measure` sets `epochs 1, ipe 50, warmup 0, anneal 0` and forces
`accum_steps: 1`, so `ms/sample` is measured on one micro-batch rather than
averaged over an accumulation loop. It also prints `batches/epoch` and refuses
configurations that would hang.

### 4.0 The global-batch rule (read before touching batch size)

For any **real** run, `batch_size × world_size × accum_steps` must equal the
paper's **128**. The lr (`4.25e-4`), warmup and WSD schedule are tuned for it —
changing the global batch changes the optimisation problem, not just the speed.

```
batch  8 × accum 16 = 128     target: fits 32GB (5090 AND the lab card)
batch  4 × accum 32 = 128     fallback if batch 8 does not fit
```

**Both do exactly 128 sample-forwards per optimizer step, and the same FLOPs.**
Larger micro-batches are faster only through reduced overhead — fewer Python
iterations, fewer kernel launches, better SM occupancy. **Expect ~1.2–1.6×, not
the 4× the accum ratio suggests.** `set_batch.py` computes `accum_steps` for you
and refuses batch sizes that do not divide 128 (except under `--measure`, where
accum is 1 and divisibility is irrelevant).

### 4.0b Verify the metrics BEFORE the long run

A metric that is computed but never logged — or logged as a constant — is worse
than no metric, because it looks like evidence.

**Offline, no GPU, run it now:**

```bash
python server/check_metrics.py --offline
```

Asserts the expressions respond correctly to known-good / known-collapsed /
known-random inputs. Must print `ALL OFFLINE CHECKS PASSED`.

**Live, on the first short run:**

```bash
python server/check_metrics.py --live sweep_s1_b8.log
```

Checks the val loader was built, that `std`/`cos`/`samp/s`/`VAL`/`retrieval`
all actually appear, and — importantly — that `std` and `cos` **vary between
steps**. A constant value means a constant is being printed, not measured.

### 4.0c Reading the metrics

| Metric | Healthy | Warning |
|---|---|---|
| `VAL` | falling, and **below 0.80** | stuck ≥0.80, or rising while train falls |
| `train − val` gap | small, stable | widening fast → overfitting; more epochs waste money |
| `std` | O(1), stable | **→ 0 = representation collapse** |
| `cos` | rising toward 1 | flat near 0 = no directional structure |
| `top1` | rising above `chance` | at chance after several epochs = not learning |
| `grad` unclipped | O(0.1–1), no spikes | persistently ≫1 = lr too high; →0 = dead |

**The L1 baselines are what make `VAL` interpretable** (verified empirically by
`--offline`, not asserted):

```
~1.13  predicting an unrelated latent   -- chance
~0.80  predicting zeros                 -- the degenerate solution
<0.80  genuinely predicting structure
```

**The trap `std` exists to catch:** a fully collapsed model scores L1 **0.746**,
which is *better* than the 0.798 zeros baseline. Loss alone reads as progress.
Only `std → 0` exposes it. **Never judge the loss without looking at `std`.**

**Judging "is training working":** by epoch 2–3 you want `VAL` falling, `std`
holding steady, and `top1` above chance. If `VAL` is flat while train falls,
you are memorising — the dataset is too small, and more compute will not fix it.

### 4.1 Stage 1 throughput and memory

Sweep **4 → 8 → 16**, then **batch 8 again as a drift control**. If the two
batch-8 numbers disagree, the box is not stable and nothing else here is
trustworthy. On 32 GB, batch 16 is expected to OOM — that is the ceiling being
recorded, not a problem.

**Do the whole sweep with one command:**

```bash
python server/sweep_memory.py --stages 1 2 --batches 4 8 16 --fits 32
```

On a 32 GB card, expect **batch 16 to OOM** (§6 predicts 34–47 GB). That is the
point: an OOM is a recorded result that establishes the ceiling, not a failure.
Each config is a separate subprocess, so one OOM does not stop the sweep. The
script also refuses to start if anything already holds the GPU, and waits for
memory to clear between configs — a crashed run leaving memory held is what
corrupted measurements on the 4080.

It runs each config for 10 optimizer steps, samples true VRAM from `nvidia-smi`
while the run is live, and prints a table ending in a **fits 32GB?** column.

Why 10 steps and not one: **Adam allocates `exp_avg`/`exp_avg_sq` lazily on the
first `optimizer.step()`** — that is precisely where stage 1 OOMed on the 16GB
box (`torch.zeros_like` in `adam.py:_init_group`). A single step never reaches
peak. Ten is plenty; a full epoch is wasted money.

Why two memory numbers:

| | what it is |
|---|---|
| `torch_alloc` | `torch.cuda.max_memory_allocated()` — PyTorch tensors only. This is train.py's `mem` |
| `nvidia_smi` | true process VRAM: adds the CUDA context (~0.3–0.5 GB), cuDNN workspaces, allocator reserved-but-unused |

**`nvidia_smi` is what decides whether a config fits a card.** `torch_alloc` runs
1–3 GB lower; planning against it will OOM you on the lab card.

An OOM in the sweep is a **result**, not a failure — it establishes the ceiling.
The script records it and carries on.

Then re-run batch 8 as a **drift control**:

```bash
python server/sweep_memory.py --batches 8 --stages 1
```

If it disagrees with the first batch-8 row, the box is unstable and nothing here
is trustworthy.

**The question this answers:** does `ms/sample` keep falling as batch grows?

- **Flattens by 8** → the 32 GB lab card gives up little. Plan around it.
- **Still falling at 32** → the lab card is genuinely handicapped, and free-but-slower
  may lose to $1/hr cloud. That is a real strategic finding worth reporting.

Record from the **last** `log_stats` line (steady state, not step 0):

| | b4 | b8 | b16 | b8 control |
|---|---|---|---|---|
| `gpu` ms | | | | |
| `mem` MB | | | | |
| `data` ms | | | | |
| **ms/sample** = `gpu / batch` | | | | |

Then pick the real-run config: the largest batch that fits **32 GB** with
headroom, and `set_batch.py` without `--measure` to restore global batch 128.

### 4.1a MANDATORY before any real run — verify the NoDDP change

`src/utils/single_gpu.py` skips `DistributedDataParallel` when `world_size == 1`.
This is why stage 2 can reach a usable batch size at all.

**Why it is safe:** at one rank every DDP operation is a no-op with itself —
all-reduce averages one value, the ÷world_size divides by 1, the broadcasts copy
to self. But DDP still allocates a contiguous gradient bucket per module at wrap
time, ~4 bytes × trainable params. In stage 2 that is ~11 GB, and **three of the
four wrapped modules are never trained**: `encoder` (not frozen, but absent from
`init_opt`'s param groups), `target_encoder` (frozen at `train.py:362`, *after*
the wrap), `dreamer_predictor` (frozen at `:375`, also after).

Freezing before the wrap does not help — PyTorch refuses to wrap a module with
no trainable parameters. `bucket_cap_mb` changes granularity, not total. There is
no "wrap but allocate nothing" mode.

The only DDP behaviour this project depends on at one GPU is the `module.` prefix
on `state_dict` keys, which `repack_stage0.py` adds and every loader expects
(HANDOFF §6). `NoDDP` reproduces that exactly.

**Gradient accumulation is unaffected** — it averages over micro-batches via the
`/ n_micro` loss scaling and autograd's accumulation into `.grad`. DDP averages
over *ranks*. Orthogonal.

**Prove it rather than believe it.** Stage 1 is deterministic (HANDOFF §5b
reproduced 0.971/0.604/0.540 bit-for-bit), so:

```bash
python server/verify_noddp.py --steps 20 --batch 8 2>&1 | tee verify_noddp.log
```

Runs stage 1 twice on an identical config — once with `NoDDP`, once with
`DJEPA_FORCE_DDP=1` restoring real DDP — and compares every per-iteration loss
from `exp/stage1/log_r0.csv` as strings.

- **PASS, all losses bit-identical** → equivalence is proven; continue.
- **FAIL** → the reasoning is wrong somewhere. `git revert` the NoDDP commit and
  run with real DDP at whatever batch fits. Do not proceed on a hunch.

Also compare `[mem: ...]` between `verify_noddp.log` and `verify_ddp.log` — the
NoDDP run should be several GB lower. Same losses, less memory, is the whole
claim.

**Re-run the sweep afterwards**, since the memory ceiling has moved:

```bash
python server/sweep_memory.py --stages 1 2 --batches 4 8 16 --fits 32
```

Stage 1 may now reach batch 16 (halving `accum_steps` to 8), and stage 2 should
reach 8 or 16 — the paper's global batch of 16 on one GPU.

### 4.1b Apply the sweep result — the config for every run after this

Read the `fits 32GB?` column. Take the **largest YES** per stage, then write the
real config once:

```bash
python server/prepare_configs.py --gpus 1 --s2-batch <largest stage-2 YES>        --epochs <from §4.6> --ipe 100 --warmup <~25%> --anneal <~30%> --save-every 4
python server/set_batch.py --batch <largest stage-1 YES>      # NO --measure
```

**Order matters:** `prepare_configs.py` reads its baseline from `git HEAD` and
rewrites the file, so `set_batch.py` must come second or its batch is discarded.
Omitting `--measure` is what makes `set_batch.py` restore `accum_steps` and the
real schedule instead of the timing stub.

**Stage 1** — `set_batch.py` derives `accum_steps` to hold global batch at the
paper's 128, and refuses sizes that do not divide it:

| batch | accum | global |
|---|---|---|
| 16 | 8 | 128 |
| **8** | **16** | **128** ← most likely |
| 4 | 32 | 128 |

**Stage 2 is different — it has NO `accum_steps` support** (`app/vjepa_2_1_dreamer_ac/train.py`
is untouched upstream), so global batch is just `batch_size × world_size`. There
is no way to reach the paper's 16 on one GPU except by fitting batch 16 outright:

| `--s2-batch` | global | vs paper (16) |
|---|---|---|
| 16 | 16 | **matches** |
| 8 | 8 | half — a deviation to state |
| 4 | 4 | quarter — state it |

The old hardcoded cap of 4 was a guess, never measured. If 16 fits, use it.

Confirm before launching:

```bash
python -c "import yaml; [print(f, yaml.safe_load(open(f))['data']['batch_size'], yaml.safe_load(open(f))['optimization'].get('accum_steps',1)) for f in ('configs/train/vjepa_2_1_dreamer_predictor.yaml','configs/train/vjepa_2_1_dreamer_ac.yaml')]"
```

Stage 1 must print `batch × accum = 128`. If a later `prepare_configs.py` run
resets stage 1's batch, re-run `set_batch.py`.

### 4.2 Does §6's memory analysis survive contact?

§6 predicts **34–47 GB at batch 16** and 24–27 GB at batch 8, from meta-tensor
saved-tensor hooks. Compare against measured `mem`.

Note `mem` is `torch.cuda.max_memory_allocated()` — PyTorch allocations only,
excluding caching-allocator overhead, so true VRAM use is somewhat higher. Read
`nvidia-smi` alongside it.

**This either validates or kills a large chunk of prior analysis. Either result
is worth reporting.**

### 4.3 Largest batch that fits 32 GB

The lab card is 32 GB. Find the batch size whose measured `mem` (plus headroom)
fits, and record its ms/sample. That is the number that actually plans the lab run.

### 4.4 Stage 2

**Stage 2 has never completed** (HANDOFF §5b — blocked by contention, not a bug).
This is the first time. Batch 2 and 4.

```bash
python server/run_stage2.sh 2>&1 | tee s2_b2.log
```

Stage 2 has **no `accum_steps` support** (`app/vjepa_2_1_dreamer_ac/train.py` is
untouched upstream), so global batch = `batch_size × world_size`.

### 4.5 Scale to the lab card

Look up both spec sheets — CUDA cores, boost clock, memory bandwidth — and
compute a ratio **RTX 5090 → RTX PRO 4500**. Both Blackwell, so this should be
reasonably honest, but it is still the largest remaining source of error in the
whole projection.

**If an RTX PRO 4500 is ever available to rent, take it over a 5090 even at a
premium** — it is the lab card exactly, so `ms/sample` measured on it needs no
scaling at all and this entire section disappears.

**Present it as an estimate with stated uncertainty.** Then:

```
lab GPU-days = (steps × ms_per_step_on_4500) / 86_400_000
steps = epochs × ipe          (ipe = 300; independent of dataset size)
```

### 4.6 Does it actually learn?

If budget remains: run stage 1 for ~500–1000 steps with a real schedule
(`warmup`/`anneal` non-zero) and watch the loss.

**Calibration for L1 in layer-normed latent space** (verify empirically, do not
trust the arithmetic):

| loss | meaning |
|---|---|
| ~1.13 | predicting an unrelated latent — chance |
| ~0.80 | predicting zeros — the trivial degenerate solution |
| below 0.80 | genuinely predicting structure |

A loss parked near 0.80 **with collapsing output variance** is representation
collapse, the classic JEPA failure mode — not convergence. Plot from
`exp/stage1/log_r0.csv` (per-iteration), not the console (running average).

---

## 4.7 Persistence — do this BEFORE launching anything long

A rented instance can be reclaimed, and your SSH can drop. Two different
problems, two different answers:

**Metrics — wandb, already wired.** `wandb.log` in train.py records `lr`, `wd`,
`grad_norm_*` and `mem` (series the CSV does not keep) and uploads live. Log in
with YOUR account and leave `WANDB_MODE` unset.

**Logs and checkpoints — pull them, from your own machine:**

```bash
# on the 4080, NOT on the rented box
bash server/pull_artifacts.sh root@<vast-host> <port>
```

Pull rather than push: vast.ai exposes an SSH host:port, while the 4080 is
behind tailscale and usually unreachable from inside a rented container.
Pulling also installs nothing on hardware you are paying for.

Logs sync every 60s (kilobytes), checkpoints every 15 min (~9 GB each). Start it
in a second terminal before the first long run and leave it going.

### Checkpointing: latest.pt is OVERWRITTEN

Both stages do the same thing (`train.py:578` / `:622`):

```python
if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):   # every epoch
    save_checkpoint(epoch + 1, latest_path)          # latest.pt -- OVERWRITTEN
    if save_every_freq > 0 and epoch % save_every_freq == 0:
        save_checkpoint(epoch + 1, f"e{epoch}.pt")   # kept
```

Upstream ships `save_every_freq: 25`. On a 12-epoch run only epoch 0 satisfies
`0 % 25 == 0`, so you keep **`e0.pt` (untrained) and nothing else** while
`latest.pt` is overwritten twelve times. Set it explicitly:

```bash
python server/prepare_configs.py --gpus 1 --epochs 12 --ipe 100        --warmup 3 --anneal 3 --save-every 4
```

That keeps `e0/e4/e8.pt` plus `latest.pt` = **~36 GB for stage 1 alone**.
`prepare_configs.py` prints the estimate. **Check `df -h` before choosing** —
vast.ai default disk allocations are often too small for this, and you also need
room for the 2.64 GB base checkpoint and the dataset.

### Why checkpoints are 9 GB, and the easy 45% saving

Each `latest.pt` is encoder **4.05 GB** + dreamer_predictor **1.70 GB** + Adam
state **3.40 GB** (`train.py:330`). With `unfreeze_vit: False` only the dreamer
is in the optimiser (`utils.py:246`), and **the encoder is frozen** — so those
4.05 GB are byte-identical every save and already sit in
`vjepa2_ac_repacked.pt`.

Stage 2 only reads `checkpoint["dreamer_predictor"]`, so a slim checkpoint would
be enough for everything downstream. **Not changed today** — it touches upstream
`train.py` and the rehearsal does not need it. Worth doing before the multi-day
lab run, where it turns a 9 GB write per epoch into 1.7 GB.

---

## 5. Before you tear down

- [ ] `s1_b8.log`, `s1_b16.log`, `s1_b8_control.log`, `s2_b2.log`, `s2_b4.log`
- [ ] `exp/stage1/log_r0.csv`, `exp/stage2/log_r0.csv`
- [ ] wandb runs synced
- [ ] `nvidia-smi` peak memory noted per config
- [ ] the filled-in table from §4.1
- [ ] **`exp/stage1/latest.pt` and `exp/stage2/latest.pt` copied off the box** —
      the instance is ephemeral. ~9 GB each; budget 15-30 min.
      To move only what is needed, extract the trained weights (stage 2 reads
      only `checkpoint["dreamer_predictor"]`, so the frozen 4.05 GB encoder and
      3.40 GB of Adam state do not need to travel):
      ```bash
      python -c "import torch; c=torch.load('exp/stage1/latest.pt',map_location='cpu',mmap=True); torch.save({'dreamer_predictor':c['dreamer_predictor'],'epoch':c['epoch']},'stage1_slim.pt')"
      ```
      1.7 GB instead of 9 GB. Keep the full file too if disk allows.

Then destroy the instance. You are billed while it exists.

---

## 6. Out of scope for this session

- **Rollout / video** — already proven end to end on the 4080 (HANDOFF §5b).
  The scripts are on this branch, but standing up CoppeliaSim on a rented box is
  a time sink for zero new information.
- **Any reportable result.** 18–200 episodes of one task is not an experiment.
- **`patch_16gb.py`** — deleted from this branch on purpose. 96 GB does not need
  bf16 hacks, and applying them would silently make the measurements describe a
  configuration the lab card will never run.

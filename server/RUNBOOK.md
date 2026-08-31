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
| Hardware | vast.ai RTX PRO 6000 Blackwell **96 GB**, ~$1/hr |
| Budget | ₹1000 ≈ $12 ≈ 12 h. **Target spend 4–5 h; keep the rest as reserve.** |
| Lab card (the real target) | RTX PRO 4500 Blackwell **32 GB**, same architecture |

96 GB means **no memory hacks**. `patch_16gb.py` is deliberately deleted from
this branch — if you find yourself wanting it, you are on the wrong branch.
Everything runs full fp32, the configuration the lab card will actually use.

### Why 96 GB when the lab card is 32 GB

Two reasons. First, it removes memory as a confound so throughput is measured
cleanly. Second, batch 16 (the paper's per-GPU batch) has **never been run** —
§6 predicts 34–47 GB from meta-tensor analysis and that has never been checked
against reality. 96 GB settles it.

You then scale to the 32 GB card by spec ratio (§4.5). That is an estimate, and
§12 of HANDOFF is unkind about estimates — label it as one.

---

## 1. GATE: you need more data before renting

**Do not rent until this is done.** Throughput measured on the current 18-episode
smoke set is invalid:

- 12 training pairs ÷ batch 16 = **0 batches** → `prepare_configs.py` exits 1
  (correctly — `drop_last=True` would hang silently, HANDOFF §6)
- batch 8 gives **1 batch/epoch** → the loader hits `StopIteration` and rebuilds
  every single step. That inflates `data` time and pollutes `iter`.

**Minimum: ~640 paired episodes in `data/train`** (20 batches/epoch at batch 32,
the largest size in the §4.1 sweep). Below ~320 you cannot test batch 32 at all.
Ideal is the full 4-task x 800-pair set, which is also the real run's dataset.

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

**2.2 Get the checkpoint there the cheap way.** You already have
`~/vjepa2_ac_repacked.pt` (**2.64 GB**) on the 4080. Copy that over tailscale
rather than re-downloading Meta's 11 GB original and re-running `repack_stage0.py`.
The 4080 is reachable on `100.119.141.93`.

```bash
scp ~/vjepa2_ac_repacked.pt <rented-box>:~/
```

If you must start from scratch: `wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`
then `python server/repack_stage0.py vjepa2-ac-vitg.pt ~/vjepa2_ac_repacked.pt`.
That checkpoint **must** carry the §5 norm rename — this branch's
`repack_stage0.py` does it; an older copy does not.

**2.3 Copy your collected data** (`data/train`, `data/val`) from the 4080 too.

**2.4 wandb.** Use **your own** account this time. (The 4080's `~/.netrc` holds
someone else's credentials — HANDOFF §5b.) `wandb login`, then leave
`WANDB_MODE` unset so the existing `wandb.log` calls actually record `lr`, `wd`,
`grad_norm_*` and `mem` — series the CSV does not keep. If the instance dies,
the data is already uploaded.

---

## 3. Box setup

```bash
git clone -b cloud-test <lab-remote> Demo-JEPA && cd Demo-JEPA
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv
```

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

**That must be empty.** A contended GPU is what made every previous measurement
worthless (HANDOFF §5b). If anything is running, stop and get a clean box.

```bash
python server/split_dataset.py
python server/prepare_configs.py --gpus 1
```

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
batch  8 × accum 16 = 128     fits 32GB   (the lab card)
batch 32 × accum  4 = 128     fits 96GB   (rented)
```

**Both do exactly 128 sample-forwards per optimizer step, and the same FLOPs.**
Larger micro-batches are faster only through reduced overhead — fewer Python
iterations, fewer kernel launches, better SM occupancy. **Expect ~1.2–1.6×, not
the 4× the accum ratio suggests.** `set_batch.py` computes `accum_steps` for you
and refuses batch sizes that do not divide 128 (except under `--measure`, where
accum is 1 and divisibility is irrelevant).

### 4.1 Stage 1 throughput and memory

Sweep **8 → 16 → 32**, then **batch 8 again as a drift control**. If the two
batch-8 numbers disagree, the box is not stable and nothing else here is
trustworthy. Add **48** (measure-only) if 32 leaves plenty of headroom.

**Do the whole sweep with one command:**

```bash
python server/sweep_memory.py --stages 1 2 --batches 8 16 32 --fits 32
```

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

| | b8 | b16 | b32 | b8 control |
|---|---|---|---|---|
| `gpu` ms | | | | |
| `mem` MB | | | | |
| `data` ms | | | | |
| **ms/sample** = `gpu / batch` | | | | |

Then pick the real-run config: the largest batch that fits **32 GB** with
headroom, and `set_batch.py` without `--measure` to restore global batch 128.

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
compute a ratio RTX PRO 6000 → RTX PRO 4500. Same Blackwell generation, so this
should be reasonably honest.

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
- [ ] **`exp/stage1/latest.pt` copied off the box** if §4.6 ran — it is the only
      trained artifact and the instance is ephemeral

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

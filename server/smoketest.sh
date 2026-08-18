#!/usr/bin/env bash
# End-to-end pipeline rehearsal, run as resumable phases.
#
# NOT a monolithic script: each phase logs separately, progress is recorded, and
# a rerun skips what already passed. Failing at phase 9 must not re-download an
# 11GB checkpoint.
#
#   bash server/smoketest.sh                  # run all pending phases
#   bash server/smoketest.sh --list           # phases + status
#   bash server/smoketest.sh --from 6         # resume at phase 6
#   bash server/smoketest.sh --only 12        # rerun one phase
#   bash server/smoketest.sh --reset          # forget all progress
#   bash server/smoketest.sh --with-sim       # include simulator phases (13-15)
#   bash server/smoketest.sh --gpus 1 --16gb  # single 16GB card (applies bf16 patches)
#
# Output: live on your terminal AND saved to .smoketest/logs/NN_name.log
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
STATE_DIR="$REPO/.smoketest"
LOG_DIR="$STATE_DIR/logs"
DONE_FILE="$STATE_DIR/completed"
mkdir -p "$LOG_DIR"
touch "$DONE_FILE"

GPUS=1; SMOKE16=0; WITH_SIM=0; FROM=0; ONLY=""; LIST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --16gb) SMOKE16=1; shift ;;
        --with-sim) WITH_SIM=1; shift ;;
        --from) FROM="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --list) LIST=1; shift ;;
        --reset) : > "$DONE_FILE"; echo "progress cleared"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-disabled}"
DEV="cuda:0"

PHASES=(
    "01 preflight            hardware, python, disk"
    "02 imports              torch + repo training modules"
    "03 dataset              extract archive and split train/val"
    "04 stage0_download      fetch V-JEPA 2.1-AC checkpoint (~11GB)"
    "05 stage0_repack        strip optimizer state, verify module. prefix"
    "06 patch16              apply 16GB bf16 patches (only with --16gb)"
    "07 configs              write smoke configs"
    "08 stage1               short stage 1 training run"
    "09 verify_stage1        latest.pt exists with expected keys"
    "10 stage2               short stage 2 training run"
    "11 deploy_ckpt          build deploy checkpoint from stage 2"
    "12 cem_debug            CEM without simulator (--debugmode True)"
    "13 sim_check            simulator env importable  [--with-sim]"
    "14 rollout              one closed-loop episode    [--with-sim]"
    "15 video                side-by-side gif           [--with-sim]"
)

if [ "$LIST" = "1" ]; then
    echo "phase  status   description"
    for p in "${PHASES[@]}"; do
        n="${p%% *}"; rest="${p#* }"
        if grep -qx "$n" "$DONE_FILE"; then st="DONE   "; else st="pending"; fi
        echo "  $n   $st  $rest"
    done
    exit 0
fi

run_phase() {
    local num="$1" name="$2"; shift 2
    local log="$LOG_DIR/${num}_${name}.log"

    if [ -n "$ONLY" ] && [ "$ONLY" != "$num" ] && [ "$ONLY" != "$((10#$num))" ]; then return 0; fi
    if [ -z "$ONLY" ] && [ "$((10#$num))" -lt "$((10#$FROM))" ]; then return 0; fi
    if [ -z "$ONLY" ] && grep -qx "$num" "$DONE_FILE"; then
        printf '  [%s] %-18s SKIP (already done)\n' "$num" "$name"; return 0
    fi

    printf '\n\033[1m=== [%s] %s ===\033[0m\n' "$num" "$name"
    local start=$SECONDS
    ( "$@" ) 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    local dur=$((SECONDS - start))

    if [ "$rc" -ne 0 ]; then
        cat <<EOF

################################################################
 FAILED at phase $num ($name) after ${dur}s, exit code $rc
 full log: $log
 last lines:
$(tail -n 20 "$log" | sed 's/^/   | /')

 Fix, then resume with:
   bash server/smoketest.sh --from $num $( [ "$SMOKE16" = 1 ] && echo -n '--16gb ' )$( [ "$WITH_SIM" = 1 ] && echo -n '--with-sim ' )--gpus $GPUS
################################################################
EOF
        exit "$rc"
    fi
    printf '  [%s] %s OK (%ss)\n' "$num" "$name" "$dur"
    grep -qx "$num" "$DONE_FILE" || echo "$num" >> "$DONE_FILE"
}

# ------------------------------------------------------------------ phases ---
p_preflight() {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || return 1
    echo "--- host ---"; free -g | head -2; df -h "$REPO" | tail -1; echo "cores: $(nproc)"
    echo "--- python ---"
    python - <<'PY'
import sys, os, re, subprocess, torch
print("interpreter :", sys.executable)
print("python      :", sys.version.split()[0])
print("torch       :", torch.__version__, "| built for CUDA", torch.version.cuda)
env = os.environ.get("CONDA_DEFAULT_ENV", "(none)")
print("conda env   :", env)
if env in ("(none)", "base"):
    print("WARNING: running in conda base. Expected a dedicated env (e.g. 'djepa').")

if torch.cuda.is_available():
    print("cuda        : OK ->", torch.cuda.get_device_name(0))
    sys.exit(0)

# Diagnose the usual cause rather than just reporting failure.
print("\nERROR: torch cannot see the GPU.")
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print("driver      :", out)
except Exception:
    out = ""
built = torch.version.cuda or "?"
print(f"""
Most likely: torch is built for CUDA {built} but the driver supports an older
runtime. A driver upgrade needs root; installing a matching torch does not.

Find your driver's max CUDA under 'CUDA Version' in plain `nvidia-smi`, then:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX

where cuXXX is that version without the dot (12.8 -> cu128). An OLDER cuXXX than
the driver supports is fine; a newer one is not. Install torch BEFORE `pip install .`
so the repo's `torch>=2` requirement does not pull the default wheel back in.
""")
sys.exit(1)
PY
}

p_imports() {
    python -c "import torch,timm,einops,h5py,decord,transformers,yaml; print('third-party OK')"
    python -c "import app.vjepa_2_1_dreamer_predictor.train, app.vjepa_2_1_dreamer_ac.train; print('repo training modules OK')"
}

p_dataset() {
    if [ ! -d "$REPO/data/rlbench_data" ]; then
        ( cd "$REPO/data" && tar -xzf rlbench_data.tar.gz )
    fi
    python server/split_dataset.py
}

p_stage0_download() {
    local f="$HOME/vjepa2-ac-vitg.pt"
    if [ -s "$f" ]; then echo "already present: $f ($(du -h "$f" | cut -f1))"; return 0; fi
    # --show-progress renders a bar only on a TTY; piped through tee it degrades
    # to one dot-line per 50KB (~220k lines for 11GB). dot:giga = 32MB/line.
    # -c resumes a partial file instead of restarting the whole download.
    wget -c --progress=dot:giga -O "$f" https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt
    echo "downloaded: $(du -h "$f" | cut -f1)"
}

p_stage0_repack() {
    python server/repack_stage0.py "$HOME/vjepa2-ac-vitg.pt" "$HOME/vjepa2_ac_repacked.pt"
}

p_patch16() {
    if [ "$SMOKE16" != "1" ]; then echo "skipped (no --16gb)"; return 0; fi
    python server/patch_16gb.py
}

p_configs() {
    python server/prepare_configs.py --gpus "$GPUS" --smoke
}

p_stage1() {
    python -m app.main --fname configs/train/vjepa_2_1_dreamer_predictor.yaml --devices "$DEV" --debugmode True
}

p_verify_stage1() {
    python - <<'PY'
import os, sys, torch
p = os.path.join("exp", "stage1", "latest.pt")
if not os.path.exists(p):
    sys.exit(f"ERROR: {p} not written -- stage 1 produced no checkpoint")
ck = torch.load(p, map_location="cpu", mmap=True)
print("keys:", sorted(ck.keys()))
for k in ("encoder", "dreamer_predictor"):
    if k not in ck:
        sys.exit(f"ERROR: '{k}' missing from stage 1 checkpoint")
print(f"size: {os.path.getsize(p)/1e9:.2f} GB   epoch: {ck.get('epoch')}   loss: {ck.get('loss')}")
PY
}

p_stage2() {
    python -m app.main --fname configs/train/vjepa_2_1_dreamer_ac.yaml --devices "$DEV" --debugmode True
}

p_deploy_ckpt() {
    mkdir -p exp
    python server/make_deploy_ckpt.py exp/stage2/latest.pt exp/stage2_deploy.pt
}

p_cem_debug() {
    python server/prepare_deploy_config.py
    python -m app.vjepa_2_1_dreamer_ac.deploy --fname configs/inference/deploy_vjepa_2_1.yaml --debugmode True
}

p_sim_check() {
    if [ "$WITH_SIM" != "1" ]; then echo "skipped (no --with-sim)"; return 0; fi
    local pysim
    pysim=$(python - <<'PY'
import re, pathlib
m = re.search(r'^PY_SIM\s*=\s*"([^"]+)"', pathlib.Path("server/run_rollout.py").read_text(), re.M)
print(m.group(1) if m else "")
PY
)
    [ -x "$pysim" ] || { echo "ERROR: PY_SIM not executable: '$pysim' -- edit server/run_rollout.py"; return 1; }
    "$pysim" -c "import pyrep, rlbench; print('sim env OK')"
    command -v Xvfb >/dev/null || { echo "ERROR: Xvfb not installed"; return 1; }
    echo "Xvfb: $(command -v Xvfb)"
}

p_rollout() {
    if [ "$WITH_SIM" != "1" ]; then echo "skipped (no --with-sim)"; return 0; fi
    python server/run_rollout.py --episodes 1 --task push_button --fresh
    ls -la rollouts/ep0 | head -5
    local n; n=$(find rollouts/ep0 -name '*.png' -o -name '*.jpg' 2>/dev/null | wc -l)
    echo "frames captured: $n"
    [ "$n" -gt 0 ] || { echo "ERROR: no frames -- check rollouts/server_ep0.log"; return 1; }
}

p_video() {
    if [ "$WITH_SIM" != "1" ]; then echo "skipped (no --with-sim)"; return 0; fi
    local ref; ref=$(find data/val -path '*/sawyer/*.hdf5' | sort | head -1)
    [ -n "$ref" ] || { echo "ERROR: no held-out sawyer episode found"; return 1; }
    python server/make_video.py --frames rollouts/ep0 --reference "$ref"
}

# -------------------------------------------------------------------- main ---
echo "repo: $REPO   gpus: $GPUS   16gb-patches: $SMOKE16   with-sim: $WITH_SIM"
echo "logs: $LOG_DIR"
START_ALL=$SECONDS

run_phase 01 preflight        p_preflight
run_phase 02 imports          p_imports
run_phase 03 dataset          p_dataset
run_phase 04 stage0_download  p_stage0_download
run_phase 05 stage0_repack    p_stage0_repack
run_phase 06 patch16          p_patch16
run_phase 07 configs          p_configs
run_phase 08 stage1           p_stage1
run_phase 09 verify_stage1    p_verify_stage1
run_phase 10 stage2           p_stage2
run_phase 11 deploy_ckpt      p_deploy_ckpt
run_phase 12 cem_debug        p_cem_debug
run_phase 13 sim_check        p_sim_check
run_phase 14 rollout          p_rollout
run_phase 15 video            p_video

cat <<EOF

================================================================
 ALL PHASES PASSED in $((SECONDS - START_ALL))s
 logs: $LOG_DIR

 Per-iteration timing (extrapolate your real run from this):
$(grep -ho 'iter: [0-9.]* ms' "$LOG_DIR/08_stage1.log" 2>/dev/null | tail -3 | sed 's/^/   /')

 The pipeline runs end to end. Record what you changed, then run the
 real configs on the big GPU.
================================================================
EOF

#!/usr/bin/env bash
set -u

export COPPELIASIM_ROOT=/mnt/log2r/CoppeliaSim
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate rlbench

export PYTHONPATH="Demo-JEPA/scripts/rlbench_tools:${PYTHONPATH:-}"
cd scripts/rlbench_tools
export PYREP_HEADLESS=1
export LIBGL_ALWAYS_SOFTWARE=1

TASK="get_ice_from_fridge"

xvfb-run -a python cli.py \
  --save_path "/mnt/log2r/RLBench/paired_data_new_${TASK}" \
  --task "${TASK}" \
  --source_robot "panda" \
  --robots panda sawyer \
  --image_size 640 480 \
  --renderer "opengl" \
  --variations -1 \
  --total_episodes 260 \
  --arm_max_velocity 1.0 \
  --arm_max_acceleration 4.0 \
  --max_demo_attempts 2 \
  --seed_master 20250811 \
  --dt 0.033 \
  --settle_pos_eps 0.01 \
  --settle_ori_eps_deg 2.0 \
  --settle_max_steps 40 \
  --camera_json "/mnt/log2r/RLBench/rlbench/scripts/cams_extrinsics.json" \
  --headless
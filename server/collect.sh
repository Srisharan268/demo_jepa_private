#!/usr/bin/env bash
# Parallel data collection on one many-core box: paired demos + play episodes.
#
#   bash server/collect.sh TASK PAIR_WORKERS PAIRS_EACH PLAY_WORKERS PLAY_EACH
#   bash server/collect.sh status              # progress, any time
#
# e.g. 48 cores:  bash server/collect.sh push_button 30 10 15 10
#   -> 300 pairs into data/collected, 150 play into data/play, and 10 held-out
#      play episodes into data/play_val (never trained on; for action_test.py).
#
# One worker = one CoppeliaSim process = ~1 core and ~1.5-2 GB RAM. Leave a
# couple of cores free. Each worker gets its own Xvfb display and its own
# seed_master, so workers never collide, and every worker skips episodes that
# already exist -- rerunning the same command RESUMES. Run inside tmux.
#
# After it finishes:
#   python server/split_dataset.py --src data/collected --val 30
#   python server/check_dataset.py
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$REPO/logs/collect"

if [ "${1:-}" = "status" ]; then
    for d in collected play play_val; do
        n=$(find "$REPO/data/$d" -name '*.hdf5' 2>/dev/null | wc -l)
        printf "%-10s %5d hdf5 files\n" "$d" "$n"
    done
    echo "workers running: $(pgrep -fc 'cli.py|play.py' || true)"
    echo "finished:        $(grep -l '\[DONE\]' "$LOGS"/*.log 2>/dev/null | wc -l) logs"
    echo "errors (last line of each log mentioning one):"
    grep -il 'Traceback\|Error' "$LOGS"/*.log 2>/dev/null | head -5 | while read -r f; do
        echo "  $(basename "$f"): $(grep -i 'error' "$f" | tail -1 | cut -c1-120)"; done
    exit 0
fi

TASK=${1:?task}; PW=${2:?pair workers}; PE=${3:?pairs per worker}
YW=${4:?play workers}; YE=${5:?play episodes per worker}

COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-$HOME/CoppeliaSim}
PY_SIM=${PY_SIM:-$(dirname "$(dirname "$(command -v conda)")")/envs/rlbench/bin/python}
[ -x "$PY_SIM" ] || { echo "PY_SIM not found ($PY_SIM); export PY_SIM=/path/to/rlbench/python" >&2; exit 1; }
[ -d "$COPPELIASIM_ROOT" ] || { echo "COPPELIASIM_ROOT not found ($COPPELIASIM_ROOT)" >&2; exit 1; }
command -v Xvfb >/dev/null || { echo "Xvfb missing (apt install xvfb)" >&2; exit 1; }
export COPPELIASIM_ROOT LD_LIBRARY_PATH="$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

mkdir -p "$LOGS"
cd "$REPO/scripts/rlbench_tools"
DISP=200

launch() {   # name, command...
    local name=$1; shift
    Xvfb ":$DISP" -screen 0 1400x900x24 -ac +extension GLX +render -noreset >/dev/null 2>&1 &
    sleep 0.5
    DISPLAY=":$DISP" "$@" > "$LOGS/$name.log" 2>&1 &
    DISP=$((DISP + 1))
}

for i in $(seq 0 $((PW - 1))); do
    launch "paired_$i" "$PY_SIM" cli.py --save_path "$REPO/data/collected" --task "$TASK" \
        --variations 1 --total_episodes "$PE" --seed_master $((100 + i)) --headless
done
for i in $(seq 0 $((YW - 1))); do
    launch "play_$i" "$PY_SIM" play.py --save_path "$REPO/data/play" --task "$TASK" \
        --episodes "$YE" --seed_master $((1 + i)) --headless
done
launch "play_val" "$PY_SIM" play.py --save_path "$REPO/data/play_val" --task "$TASK" \
    --episodes 10 --seed_master 999 --headless

echo "launched $PW paired + $YW play + 1 play_val workers; logs in $LOGS"
echo "progress: bash server/collect.sh status"
wait
echo "all workers exited"; bash "$0" status

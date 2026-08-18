#!/usr/bin/env bash
# Restore the prebuilt simulator environment from rlbench_env.tar.gz.
#
# Faster than building from source (minutes vs an hour), and it is the exact
# environment that already ran end to end:
#
#   CoppeliaSim 4.1 (Qt 5.12.5 / boost 1.71.0 -- Ubuntu 20.04 build)
#   PyRep 4.1.0.3   RLBench 1.2.0   Python 3.10
#   numpy 2.2.6  scipy 1.15.3  h5py 3.16.0  cffi 1.14.2  pyquaternion 0.9.9
#
# IMPORTANT: the archive stores ABSOLUTE paths -- `opt/conda/envs/rlbench` and
# `content/CoppeliaSim`. Do NOT extract with `-C /`: that needs root and would
# write straight into /opt/conda, likely clobbering the server's own conda.
# This script extracts into a user-writable prefix instead and points the env
# vars at the relocated paths.
#
# Usage:
#   bash server/use_prebuilt_sim.sh ~/rlbench_env.tar.gz [install_prefix]
set -euo pipefail

TARBALL="${1:?usage: use_prebuilt_sim.sh <rlbench_env.tar.gz> [prefix]}"
PREFIX="${2:-$HOME/simenv}"

[ -f "$TARBALL" ] || { echo "ERROR: not found: $TARBALL" >&2; exit 1; }

echo "=== host check ==="
. /etc/os-release 2>/dev/null || true
echo "OS    : ${PRETTY_NAME:-unknown}"
echo "glibc : $(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo '?')"
echo
echo "The archive was built on Ubuntu 22.04 (Colab). Binaries need glibc >= 2.31."
echo "On an older host (CentOS 7, Ubuntu 18.04) they will not run -- use"
echo "server/install_sim_env.sh instead."
echo

mkdir -p "$PREFIX"
echo "=== extracting (~558MB, a few minutes) ==="
tar -xzf "$TARBALL" -C "$PREFIX"

SIM_ROOT="$PREFIX/content/CoppeliaSim"
SIM_PY="$PREFIX/opt/conda/envs/rlbench/bin/python"

for p in "$SIM_ROOT" "$SIM_PY"; do
    [ -e "$p" ] || { echo "ERROR: expected $p after extraction" >&2; exit 1; }
done
[ -f "$SIM_ROOT/coppeliaSim.sh" ] || { echo "ERROR: $SIM_ROOT/coppeliaSim.sh missing" >&2; exit 1; }
chmod +x "$SIM_ROOT/coppeliaSim.sh" "$SIM_PY" 2>/dev/null || true

export COPPELIASIM_ROOT="$SIM_ROOT"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

if ! grep -q "COPPELIASIM_ROOT" ~/.bashrc 2>/dev/null; then
    {
        echo ""
        echo "# CoppeliaSim / PyRep (added by Demo-JEPA server/use_prebuilt_sim.sh)"
        echo "export COPPELIASIM_ROOT=$SIM_ROOT"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT'
        echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT'
    } >> ~/.bashrc
    echo "env vars appended to ~/.bashrc"
fi

echo
echo "=== verify ==="
# A relocated conda env normally still works when the interpreter is invoked
# directly (python finds its stdlib relative to the binary). Console scripts in
# bin/ keep their old shebang and will not -- we only ever call the binary.
if "$SIM_PY" -c "import pyrep, rlbench; print('sim env OK (relocated)')"; then
    echo
    echo "-------------------------------------------------------------"
    echo "Paste into the top of server/run_rollout.py:"
    echo "  PY_SIM = \"$SIM_PY\""
    echo "  COPPELIASIM_ROOT = \"$SIM_ROOT\""
    echo "-------------------------------------------------------------"
    echo
    echo "Then confirm CoppeliaSim starts headless:"
    echo "  Xvfb :99 -screen 0 1400x900x24 -ac -noreset > /dev/null 2>&1 &"
    echo "  DISPLAY=:99 $SIM_ROOT/coppeliaSim.sh -h -q & sleep 15; pkill -f coppeliaSim"
else
    echo
    echo "FAILED to import from the relocated env." >&2
    echo "Most likely causes, in order:" >&2
    echo "  1. Missing system libs -- run: bash server/install_sim_env.sh --skip-apt" >&2
    echo "     (or just the apt section of it) to pull in the X11/mesa packages." >&2
    echo "  2. libffi7 absent. CoppeliaSim 4.1 links libffi.so.7; 22.04+ ships libffi8." >&2
    echo "  3. glibc too old on this host." >&2
    echo "If none apply, fall back to: bash server/install_sim_env.sh" >&2
    exit 1
fi

#!/usr/bin/env bash
# Install CoppeliaSim + PyRep + RLBench into a separate conda env.
#
# This env holds NO torch. deploy.py runs in the training env (djepa) and talks
# to server.py over localhost:9001, so the two dependency stacks never meet.
#
# Provenance:
#   - apt packages, libffi7, and the env vars are exactly what the reference
#     Colab runtime needed (notebook cell 23). Those are the non-obvious bits.
#   - CoppeliaSim 4.1 + PyRep + RLBench install follows the upstream PyRep and
#     RLBench READMEs. The notebook itself restored a prebuilt tarball rather
#     than installing, so this part is reconstructed, not copied.
#
# CoppeliaSim 4.1 links libffi.so.7. Ubuntu 22.04+ ships libffi8, which is why
# the libffi7 .deb below is needed -- without it CoppeliaSim exits immediately
# with a library error that does not mention libffi.
#
# Usage:
#   bash server/install_sim_env.sh                 # full install
#   bash server/install_sim_env.sh --skip-apt      # if you lack sudo
set -euo pipefail

SIM_ENV="${SIM_ENV:-rlbench}"
SIM_ROOT="${SIM_ROOT:-$HOME/CoppeliaSim}"
COPPELIA_URL="${COPPELIA_URL:-https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz}"
SKIP_APT=0
[ "${1:-}" = "--skip-apt" ] && SKIP_APT=1

echo "=== 0. environment ==="
. /etc/os-release 2>/dev/null || true
echo "OS       : ${PRETTY_NAME:-unknown}"
echo "sim env  : $SIM_ENV"
echo "sim root : $SIM_ROOT"

# ---------------------------------------------------------------- 1. apt ---
if [ "$SKIP_APT" = "0" ]; then
    echo
    echo "=== 1. system libraries (needs sudo) ==="
    # Exactly the set the Colab runtime required. Xvfb is mandatory: CoppeliaSim
    # will not start without a display even in --headless mode.
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        xvfb x11-utils \
        libgl1-mesa-glx libglu1-mesa libosmesa6 \
        libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \
        libxslt1.1 libsm6 libxext6 \
        wget xz-utils build-essential

    echo
    echo "=== 2. libffi7 (CoppeliaSim 4.1 needs it; 22.04+ ships libffi8) ==="
    if ldconfig -p | grep -q 'libffi\.so\.7'; then
        echo "libffi7 already present"
    else
        tmp=$(mktemp -d)
        wget -q -O "$tmp/libffi7.deb" \
            http://archive.ubuntu.com/ubuntu/pool/main/libf/libffi/libffi7_3.3-4_amd64.deb
        sudo dpkg -i "$tmp/libffi7.deb" || sudo apt-get -f install -y
        rm -rf "$tmp"
    fi
else
    echo
    echo "=== 1-2. SKIPPED (--skip-apt) ==="
    echo "Without sudo, ask your admin for the package list in this script, or"
    echo "install the X11/mesa libs from conda-forge into the $SIM_ENV env."
fi

# -------------------------------------------------------- 3. CoppeliaSim ---
echo
echo "=== 3. CoppeliaSim ==="
if [ -d "$SIM_ROOT" ]; then
    echo "already at $SIM_ROOT"
else
    tmp=$(mktemp -d)
    echo "downloading (~400MB)..."
    wget -q --show-progress -O "$tmp/coppelia.tar.xz" "$COPPELIA_URL"
    mkdir -p "$SIM_ROOT"
    tar -xf "$tmp/coppelia.tar.xz" -C "$SIM_ROOT" --strip-components=1
    rm -rf "$tmp"
    echo "extracted to $SIM_ROOT"
fi
[ -f "$SIM_ROOT/coppeliaSim.sh" ] || {
    echo "ERROR: $SIM_ROOT/coppeliaSim.sh missing -- extraction layout unexpected." >&2
    echo "Check the archive; CoppeliaSim download URLs change between releases." >&2
    exit 1
}

# ----------------------------------------------------------- 4. env vars ---
# PyRep reads all three at build AND run time. Exported here for the build
# below, and appended to ~/.bashrc so server.py sees them later.
export COPPELIASIM_ROOT="$SIM_ROOT"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

if ! grep -q "COPPELIASIM_ROOT" ~/.bashrc 2>/dev/null; then
    {
        echo ""
        echo "# CoppeliaSim / PyRep (added by Demo-JEPA server/install_sim_env.sh)"
        echo "export COPPELIASIM_ROOT=$SIM_ROOT"
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT'
        echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT'
    } >> ~/.bashrc
    echo "env vars appended to ~/.bashrc"
fi

# ------------------------------------------------ 5. conda env + PyRep -----
echo
echo "=== 5. conda env '$SIM_ENV' (python 3.10, no torch) ==="
eval "$(conda shell.bash hook)"
conda env list | grep -q "^$SIM_ENV " || conda create -n "$SIM_ENV" python=3.10 -y
conda activate "$SIM_ENV"

pip install -q --upgrade pip
pip install -q numpy cffi

echo
echo "=== 6. PyRep ==="
# Built against COPPELIASIM_ROOT, so the exports above must be set first.
pip install -q git+https://github.com/stepjam/PyRep.git

echo
echo "=== 7. RLBench ==="
pip install -q git+https://github.com/stepjam/RLBench.git

# The server also needs these for the socket protocol and image saving.
pip install -q h5py pillow scipy

# ------------------------------------------------------------ 8. verify ---
echo
echo "=== 8. verify ==="
python -c "import pyrep, rlbench; print('sim env OK')"

echo
echo "-------------------------------------------------------------"
echo "Paste these into the top of server/run_rollout.py:"
echo "  PY_SIM = \"$(which python)\""
echo "  COPPELIASIM_ROOT = \"$SIM_ROOT\""
echo "-------------------------------------------------------------"
echo
echo "Next: verify headless rendering actually works (runbook step 16):"
echo "  Xvfb :99 -screen 0 1400x900x24 -ac -noreset > /dev/null 2>&1 &"
echo "  DISPLAY=:99 $SIM_ROOT/coppeliaSim.sh -h -q &"
echo "  sleep 15 && pkill -f coppeliaSim && echo 'CoppeliaSim launched headless OK'"

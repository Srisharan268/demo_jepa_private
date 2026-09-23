# Demo-JEPA lab image: two isolated environments, nothing else.
#
#   djepa    python 3.12 + torch (cu128)   training, deploy, diagnostics
#   rlbench  python 3.10 + PyRep + RLBench  CoppeliaSim server, data collection
#
# The REPO IS NOT COPIED IN. Mount it at run time (docker/run.sh does this), so
# code edits need no rebuild and data/checkpoints/outputs stay on the host.
#
# Build from the repo root (.dockerignore keeps data and checkpoints out of the
# build context -- without it docker uploads ~12 GB before starting):
#   docker build -f docker/Dockerfile -t demojepa .

# Blackwell cards (RTX 5090 etc., sm_120) need CUDA >= 12.8 and torch >= 2.7.
# If this tag is unavailable, any nvidia/cuda 12.8+ ubuntu22.04 runtime tag works:
#   docker build --build-arg BASE=<tag> ...
ARG BASE=nvidia/cuda:12.8.1-runtime-ubuntu22.04
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo wget ca-certificates git bzip2 xz-utils build-essential tmux \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q -O /tmp/mf.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/mf.sh -b -p /opt/conda && rm /tmp/mf.sh
ENV PATH=/opt/conda/bin:$PATH

# ---- training env ----------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN conda create -n djepa python=3.12 -y \
    && /opt/conda/envs/djepa/bin/pip install --no-cache-dir torch torchvision \
         --index-url https://download.pytorch.org/whl/cu128 \
    && /opt/conda/envs/djepa/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# ---- simulator env: the SAME script used on a bare host --------------------
# HOME is redirected so the script's ~/.bashrc append (which exports CoppeliaSim
# onto LD_LIBRARY_PATH) lands in a throwaway file. CoppeliaSim ships its own Qt
# and other shared libraries; on the global library path they can shadow the
# system's for the torch process. run_rollout.py adds it for the sim only.
COPY server/install_sim_env.sh /tmp/install_sim_env.sh
RUN HOME=/tmp/simbuild SIM_ENV=rlbench SIM_ROOT=/opt/CoppeliaSim \
        bash /tmp/install_sim_env.sh \
    && rm -rf /tmp/simbuild /root/.cache \
    && chmod -R a+rwX /opt/CoppeliaSim
# ^ docker/run.sh runs as your uid, not root, and CoppeliaSim writes its user
#   settings into its own install directory.

# Paths run_rollout.py and recover_rng.py read. NOT LD_LIBRARY_PATH -- see above.
ENV COPPELIASIM_ROOT=/opt/CoppeliaSim \
    PY_SIM=/opt/conda/envs/rlbench/bin/python \
    PATH=/opt/conda/envs/djepa/bin:/opt/conda/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/Demo-JEPA
CMD ["bash"]

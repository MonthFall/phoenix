#!/usr/bin/env bash
# Convenience script to install/reinstall phxloader_v1 in conda environment.
# Prerequisite: conda activate /mnt/nvme4/dataset/conda_envs/phoenix
# Usage: bash install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Require conda environment ──
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: No conda environment active."
    echo "  Run: conda activate /mnt/nvme4/dataset/conda_envs/phoenix"
    exit 1
fi

# ── Environment setup (LD_LIBRARY_PATH and GCC not auto-set if invoked via bash install.sh without activate) ──
export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -f /opt/rh/gcc-toolset-12/enable ]; then
    source /opt/rh/gcc-toolset-12/enable
fi

# ── Install ──
echo ">>> Installing phxloader_v1 to ${CONDA_PREFIX} (no-deps, no-build-isolation)..."
python -m pip install . \
    --no-deps \
    --no-build-isolation \
    --force-reinstall

# ── Verify (-P flag prevents current dir from shadowing installed package) ──
echo ">>> Verifying import..."
python -P -c 'from phxloader_v1 import PhxLoaderV1; print("phxloader_v1 import OK")'

echo ">>> Verifying torch unchanged..."
python -P -c 'import torch; print(f"torch=={torch.__version__}, cuda={torch.cuda.is_available()}")'

echo ">>> Done."

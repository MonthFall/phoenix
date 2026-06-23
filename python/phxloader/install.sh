#!/usr/bin/env bash
# Convenience script to install/reinstall phxloader without affecting torch.
# Usage: bash install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_PACKAGES="/mnt/nvme4/dataset/pip_workspace/site-packages"
TMPDIR="/mnt/nvme4/dataset/pip_workspace/tmp"

export TMPDIR
export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Enable newer GCC if available
if [ -f /opt/rh/gcc-toolset-12/enable ]; then
    source /opt/rh/gcc-toolset-12/enable
fi

echo ">>> Installing phxloader (no-deps, target=$SITE_PACKAGES)..."
python3.11 -m pip install . \
    --target "$SITE_PACKAGES" \
    --upgrade \
    --no-deps

echo ">>> Verifying import..."
PYTHONPATH="/data/home/yueluochen/phoenix_for_vllm/vllm:$SITE_PACKAGES" \
    python3.11 -c "from phxloader import PhxLoader; print('phxloader import OK')"

echo ">>> Verifying torch unchanged..."
PYTHONPATH="$SITE_PACKAGES" \
    python3.11 -c "import torch; print(f'torch=={torch.__version__}, cuda={torch.cuda.is_available()}')"

echo ">>> Done."

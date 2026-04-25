#!/usr/bin/env bash
# bootstrap.sh -- run ONCE on a fresh AutoDL instance to set up the env.
# Usage on the AutoDL machine:
#   bash bootstrap.sh
#
# Assumptions:
#   * Image: PyTorch 2.x + CUDA 12.x (any AutoDL official PyTorch image works)
#   * Repo synced to /root/mujoco_rl_fuzz/  via tools/autodl/sync_to_remote.sh
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root
echo "[bootstrap] cwd=$PWD"

# 1. system libs MuJoCo needs (headless rendering deps optional)
apt-get update -y
apt-get install -y --no-install-recommends \
    libgl1 libosmesa6 libglfw3 libegl1 git rsync || true

# 2. python deps
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-rl.txt   # torch already in image; this is a no-op upgrade
pip install -r requirements-llm.txt

# 3. quick smoke test
python demo_smoke.py | tail -n 5
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "[bootstrap] DONE."

#!/bin/bash
# =============================================================================
# Google Colab SUMO RL Setup Script
# =============================================================================
# Használat Colab-ban:
#   !git clone https://github.com/wagnertamas/intergreen_matrix_calculator.git
#   %cd intergreen_matrix_calculator
#   !chmod +x colab_setup.sh && ./colab_setup.sh
# =============================================================================

set -e

echo "============================================"
echo "  SUMO RL — Google Colab Setup"
echo "============================================"

# --- 1. SUMO telepítés ---
echo "[1/4] SUMO telepítés..."
apt-get update -qq
apt-get install -y -qq software-properties-common > /dev/null 2>&1
add-apt-repository ppa:sumo/stable -y > /dev/null 2>&1
apt-get update -qq
apt-get install -y -qq sumo sumo-tools sumo-doc > /dev/null 2>&1

export SUMO_HOME=/usr/share/sumo
echo "export SUMO_HOME=/usr/share/sumo" >> ~/.bashrc

echo "  ✓ SUMO $(sumo --version 2>&1 | head -1)"

# --- 2. Python csomagok ---
echo "[2/4] Python csomagok telepítése..."
pip install -q --upgrade pip
pip install -q \
    numpy pandas matplotlib scipy \
    gymnasium stable-baselines3 sb3-contrib "shimmy>=2.0.0" \
    tensorboard torch torchvision \
    onnx onnxscript \
    pyyaml wandb \
    libsumo libtraci sumolib

echo "  ✓ Python csomagok kész"

# --- 3. Környezeti változók ---
echo "[3/4] Környezet beállítása..."
export USE_LIBSUMO=1
echo "export USE_LIBSUMO=1" >> ~/.bashrc

# --- 4. Gyors ellenőrzés ---
echo "[4/4] Rendszer ellenőrzés..."
python -c "
import sys
print(f'  Python: {sys.version.split()[0]}')

import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"})')

import stable_baselines3 as sb3
print(f'  SB3: {sb3.__version__}')

from sb3_contrib import QRDQN
print(f'  QR-DQN: ✓')

import gymnasium
print(f'  Gymnasium: {gymnasium.__version__}')

try:
    import libtraci
    print(f'  libtraci: ✓ (preferred)')
except ImportError:
    print(f'  libtraci: ✗')

try:
    import libsumo
    print(f'  libsumo: ✓')
except ImportError:
    print(f'  libsumo: ✗ (traci fallback)')

import wandb
print(f'  WandB: {wandb.__version__}')

import os
sumo_home = os.environ.get('SUMO_HOME', 'NOT SET')
print(f'  SUMO_HOME: {sumo_home}')
"

echo ""
echo "============================================"
echo "  ✓ Setup kész! Következő lépések:"
echo "============================================"
echo ""
echo "  # 1. WandB login (API key szükséges):"
echo "  import wandb"
echo "  wandb.login()"
echo ""
echo "  # 2. Tanítás indítása:"
echo "  !USE_LIBSUMO=1 python main_headless.py --config training_config.yaml --timesteps 50000"
echo ""
echo "  # 3. VAGY WandB sweep:"
echo "  !wandb sweep data/sweep_config.yaml --project sumo-rl-sweep"
echo "  !USE_LIBSUMO=1 wandb agent <entity/project/sweep_id>"
echo ""

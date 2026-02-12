#!/bin/bash
set -euo pipefail

# ============================================================
# BraTS test job for HTCondor
# Runs: baseline, 1 epoch, seed=42
# ============================================================

PROJECT_DIR="/afs/cern.ch/work/m/mmolinav/public/SEG_DP_KFC"
BRATS_ROOT="/eos/project/d/diagbox/BRATS2025/Glioma_seg_pre_post_treatment_mri/PRE/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS2025-GLI-PRE-Challenge-TrainingData"

echo "=== Job started: $(date) ==="
echo "Hostname: $(hostname)"
echo "GPU info:"
nvidia-smi || echo "nvidia-smi not available"

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r "${PROJECT_DIR}/requirements.txt"

# Add project to PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Run BraTS training (test: 1 epoch, baseline)
python "${PROJECT_DIR}/scripts/train_brats.py" \
    --method baseline \
    --epochs 1 \
    --lr 0.001 \
    --seed 42 \
    --batch_size 8 \
    --patch_size 64 \
    --data_root "${BRATS_ROOT}" \
    --output_dir "${PROJECT_DIR}/outputs" \
    --device cuda

echo "=== Job finished: $(date) ==="

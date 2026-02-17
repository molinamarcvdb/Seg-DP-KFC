#!/bin/bash
set -euo pipefail

# ============================================================
# Quick test: 1 epoch baseline, verifies venv + data + GPU work
# ============================================================

PROJECT_DIR="/afs/cern.ch/work/m/mmolinav/public/SEG_DP_KFC"
HEPATIC_ROOT="/eos/project/d/diagbox/MSD/Task08_HepaticVessel"

echo "=== Test job started: $(date) ==="
echo "Hostname: $(hostname)"
nvidia-smi || echo "nvidia-smi not available"

# Use CVMFS Python 3.11 (system python is 3.9, packages need >=3.10)
LCG_PYTHON="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/bin/python3"
${LCG_PYTHON} -m venv venv
source venv/bin/activate
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r "${PROJECT_DIR}/requirements.txt"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python3 "${PROJECT_DIR}/scripts/train_hepatic.py" \
    --method baseline \
    --epochs 1 \
    --lr 0.001 \
    --optimizer adam \
    --seed 42 \
    --batch_size 4 \
    --patch_size 128 \
    --patches_per_volume 4 \
    --target all \
    --data_root "${HEPATIC_ROOT}" \
    --output_dir "${PROJECT_DIR}/outputs" \
    --device cuda \
    --num_workers 2 \
    --eval_volumes 2

echo "=== Test job finished: $(date) ==="

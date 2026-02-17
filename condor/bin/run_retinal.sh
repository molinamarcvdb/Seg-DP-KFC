#!/bin/bash
set -euo pipefail

# ============================================================
# Retinal Vessel Sample Efficiency job for HTCondor
# Arguments: method epsilon seed subset_fraction
# ============================================================

METHOD="${1}"
EPSILON="${2}"
SEED="${3}"
SUBSET_FRACTION="${4}"

PROJECT_DIR="/afs/cern.ch/work/m/mmolinav/public/SEG_DP_KFC"
DATA_ROOT="/eos/project-d/diagbox/datasets/retinal_vessel"

echo "=== Job started: $(date) ==="
echo "Hostname: $(hostname)"
echo "Method: ${METHOD}, Epsilon: ${EPSILON}, Seed: ${SEED}, Fraction: ${SUBSET_FRACTION}"
echo "GPU info:"
nvidia-smi || echo "nvidia-smi not available"

# Use CVMFS Python 3.11
LCG_PYTHON="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/bin/python3"
${LCG_PYTHON} -m venv venv
source venv/bin/activate
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r "${PROJECT_DIR}/requirements.txt"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Build command
CMD="python3 ${PROJECT_DIR}/scripts/train_fundus.py \
    --method ${METHOD} \
    --epochs 30 \
    --lr 0.001 \
    --optimizer sgd \
    --seed ${SEED} \
    --batch_size 16 \
    --image_size 128 \
    --subset_fraction ${SUBSET_FRACTION} \
    --data_root ${DATA_ROOT} \
    --output_dir ${PROJECT_DIR}/outputs \
    --device cuda"

# Add epsilon for DP methods
if [ "${METHOD}" != "baseline" ]; then
    CMD="${CMD} --epsilon ${EPSILON}"
fi

echo "Running: ${CMD}"
eval ${CMD}

echo "=== Job finished: $(date) ==="

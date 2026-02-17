#!/bin/bash
set -euo pipefail

# ============================================================
# Hepatic Vessel Segmentation job for HTCondor
# Arguments: method epsilon seed lr optimizer
# ============================================================

METHOD="${1}"
EPSILON="${2}"
SEED="${3}"
LR="${4}"
OPTIMIZER="${5}"

PROJECT_DIR="/afs/cern.ch/work/m/mmolinav/public/SEG_DP_KFC"
HEPATIC_ROOT="/eos/project/d/diagbox/MSD/Task08_HepaticVessel"

echo "=== Job started: $(date) ==="
echo "Hostname: $(hostname)"
echo "Method: ${METHOD}, Epsilon: ${EPSILON}, Seed: ${SEED}, LR: ${LR}, Optimizer: ${OPTIMIZER}"
echo "GPU info:"
nvidia-smi || echo "nvidia-smi not available"

# Use CVMFS Python 3.11 (system python is 3.9, packages need >=3.10)
LCG_PYTHON="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/bin/python3"
${LCG_PYTHON} -m venv venv
source venv/bin/activate
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r "${PROJECT_DIR}/requirements.txt"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# DP methods need smaller batch size (Opacus unfold3d is very memory hungry)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "${METHOD}" == "baseline" ]; then
    BATCH_SIZE=4
else
    BATCH_SIZE=1
fi

# Build command
CMD="python3 ${PROJECT_DIR}/scripts/train_hepatic.py \
    --method ${METHOD} \
    --epochs 50 \
    --lr ${LR} \
    --optimizer ${OPTIMIZER} \
    --seed ${SEED} \
    --batch_size ${BATCH_SIZE} \
    --patch_size 96 \
    --patches_per_volume 16 \
    --target all \
    --data_root ${HEPATIC_ROOT} \
    --output_dir /afs/cern.ch/work/m/mmolinav/public/SEG_DP_KFC/outputs \
    --device cuda \
    --num_workers 2"

# Add epsilon for DP methods
if [ "${METHOD}" != "baseline" ]; then
    CMD="${CMD} --epsilon ${EPSILON}"
fi

echo "Running: ${CMD}"
eval ${CMD}

echo "=== Job finished: $(date) ==="

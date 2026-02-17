#!/bin/bash
# Resume BraTS experiments from where we left off
# 
# COMPLETED:
#   - dp_synthetic seed 456 ✓
#
# REMAINING (9 runs):
#   - dp_public seeds 123, 456 (2 runs)
#   - dp_shampoo seeds 42, 123, 456 (3 runs)
#   - dp_shampoo_synth seeds 42, 123, 456 (3 runs)
#   - baseline seeds 123, 456 (1 run - seed 42 already done)
#
# BraTS: 10 epochs, batch=8, patch=64, ~25-30 min/run
# Estimated total: ~4-5 hours
#
# Usage: nohup bash scripts/resume_brats.sh > logs/brats_resume.log 2>&1 &
#
# NOTE: Run max 2 parallel jobs on A100 (3 causes OOM with Shampoo)

set -e
cd /home/mmolinav/Projects/Seg-DP-KFC/
mkdir -p logs

EPOCHS=10
BATCH=8
PATCH=64
WORKERS=4
EPSILON=8.0
DEVICE=cuda

run_experiment() {
    local method=$1
    local seed=$2
    local lr=$3
    
    echo ""
    echo "============================================"
    echo "Running: method=$method seed=$seed lr=$lr"
    echo "Started: $(date)"
    echo "============================================"
    
    uv run python scripts/train_brats.py \
        --method $method \
        --epochs $EPOCHS \
        --lr $lr \
        --seed $seed \
        --batch_size $BATCH \
        --device $DEVICE \
        --num_workers $WORKERS \
        --patch_size $PATCH \
        --epsilon $EPSILON
    
    echo "Finished: $(date)"
    echo ""
}

echo "============================================"
echo "RESUMING BraTS experiments: $(date)"
echo "============================================"

# 1. dp_public seeds 123, 456
echo ""
echo ">>> dp_public seeds 123, 456"
run_experiment dp_public 123 0.01
run_experiment dp_public 456 0.01

# 2. dp_shampoo seeds 42, 123, 456
echo ""
echo ">>> dp_shampoo seeds 42, 123, 456"
run_experiment dp_shampoo 42 0.01
run_experiment dp_shampoo 123 0.01
run_experiment dp_shampoo 456 0.01

# 3. dp_shampoo_synth seeds 42, 123, 456
echo ""
echo ">>> dp_shampoo_synth seeds 42, 123, 456"
run_experiment dp_shampoo_synth 42 0.01
run_experiment dp_shampoo_synth 123 0.01
run_experiment dp_shampoo_synth 456 0.01

# 4. Re-run baseline seeds 123, 456 (with threshold fix)
echo ""
echo ">>> baseline seeds 123, 456 (re-run with threshold fix)"
run_experiment baseline 123 0.001
run_experiment baseline 456 0.001

echo ""
echo "============================================"
echo "All REMAINING BraTS experiments finished: $(date)"
echo "============================================"
echo ""
echo "Run to analyze: uv run python scripts/analyze_results.py"

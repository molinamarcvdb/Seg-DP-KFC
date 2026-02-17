#!/bin/bash
# Resume BraTS experiments with safer multiprocessing settings
# 
# COMPLETED:
#   - dp_synthetic seed 42 ✓
#   - dp_synthetic seed 456 ✓
#   - dp_public seed 123 ✓
#
# REMAINING (8 runs):
#   - dp_public seed 456 (1 run)
#   - dp_shampoo seeds 42, 123, 456 (3 runs)
#   - dp_shampoo_synth seeds 42, 123, 456 (3 runs)
#   - baseline seed 123 (1 run)
#
# BraTS: 10 epochs, batch=8, patch=64, ~25-30 min/run
# Estimated total: ~3.5-4 hours
#
# Usage: nohup bash scripts/resume_brats_safe.sh > logs/brats_safe.log 2>&1 &

set -e
cd /home/mmolinav/Projects/Seg-DP-KFC/
mkdir -p logs

EPOCHS=10
BATCH=8
PATCH=64
WORKERS=2  # Reduced from 4 to avoid loky crashes
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
    
    # Run with retry on failure
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt of $max_attempts..."
        
        if uv run python scripts/train_brats.py \
            --method $method \
            --epochs $EPOCHS \
            --lr $lr \
            --seed $seed \
            --batch_size $BATCH \
            --device $DEVICE \
            --num_workers $WORKERS \
            --patch_size $PATCH \
            --epsilon $EPSILON; then
            echo "✓ Success on attempt $attempt"
            break
        else
            echo "✗ Failed on attempt $attempt"
            if [ $attempt -lt $max_attempts ]; then
                echo "Waiting 30s before retry..."
                sleep 30
            fi
            attempt=$((attempt + 1))
        fi
    done
    
    if [ $attempt -gt $max_attempts ]; then
        echo "ERROR: Failed after $max_attempts attempts"
        # Continue with next experiment instead of exiting
    fi
    
    echo "Finished: $(date)"
    echo ""
}

echo "============================================"
echo "RESUMING BraTS experiments (SAFE MODE): $(date)"
echo "============================================"

# 1. dp_public seed 456
echo ""
echo ">>> dp_public seed 456"
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

# 4. Re-run baseline seed 123 (with threshold fix)
echo ""
echo ">>> baseline seed 123 (re-run with threshold fix)"
run_experiment baseline 123 0.001

echo ""
echo "============================================"
echo "All REMAINING BraTS experiments finished: $(date)"
echo "============================================"
echo ""
echo "Run to analyze: uv run python scripts/analyze_results.py"

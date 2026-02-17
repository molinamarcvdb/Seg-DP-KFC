#!/bin/bash
# Run epsilon sweep for BraTS 2025
# 
# Sweep configuration:
#   - Epsilon values: 1.0, 2.0, 4.0, 8.0
#   - Methods: dp_synthetic, dp_shampoo_synth
#   - Seeds: 42, 123, 456
#   - Total: 20 missing runs (4 completed, 20 to go)
#
# Estimated time: ~20 runs × 30 min = ~10 hours
#
# Usage: nohup bash scripts/run_brats_epsilon_sweep.sh > logs/brats_epsilon_sweep.log 2>&1 &

set -e
cd /home/mmolinav/Projects/Seg-DP-KFC/
mkdir -p logs

EPOCHS=10
BATCH=8
PATCH=64
WORKERS=2  # Reduced to avoid crashes
DEVICE=cuda
LR=0.01  # Tuned learning rate for BraTS DP methods

run_experiment() {
    local method=$1
    local epsilon=$2
    local seed=$3
    
    echo ""
    echo "============================================"
    echo "Running: method=$method eps=$epsilon seed=$seed lr=$LR"
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
            --lr $LR \
            --seed $seed \
            --batch_size $BATCH \
            --device $DEVICE \
            --num_workers $WORKERS \
            --patch_size $PATCH \
            --epsilon $epsilon; then
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
        echo "ERROR: Failed after $max_attempts attempts - continuing with next experiment"
    fi
    
    echo "Finished: $(date)"
    echo ""
}

echo "============================================"
echo "BraTS Epsilon Sweep: $(date)"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Methods: dp_synthetic, dp_shampoo_synth"
echo "  Epsilons: 1.0, 2.0, 4.0, 8.0"
echo "  Seeds: 42, 123, 456"
echo "  Total runs needed: 20 (4 already done)"
echo ""

# Epsilon sweep for dp_synthetic (all epsilons, all seeds)
echo ""
echo "========================================"
echo "METHOD: dp_synthetic"
echo "========================================"

for epsilon in 1.0 2.0 4.0 8.0; do
    echo ""
    echo ">>> Epsilon = $epsilon"
    
    for seed in 42 123 456; do
        # Skip seed 456, eps=8.0 (already done)
        if [ "$epsilon" = "8.0" ] && [ "$seed" = "456" ]; then
            echo "Skipping eps=$epsilon seed=$seed (already completed)"
            continue
        fi
        
        run_experiment dp_synthetic $epsilon $seed
    done
done

# Epsilon sweep for dp_shampoo_synth (eps 1,2,4 only - 8.0 all seeds done)
echo ""
echo "========================================"
echo "METHOD: dp_shampoo_synth"
echo "========================================"

for epsilon in 1.0 2.0 4.0; do
    echo ""
    echo ">>> Epsilon = $epsilon"
    
    for seed in 42 123 456; do
        run_experiment dp_shampoo_synth $epsilon $seed
    done
done

echo ""
echo "============================================"
echo "Epsilon sweep completed: $(date)"
echo "============================================"
echo ""
echo "Run to analyze: uv run python scripts/analyze_results.py"

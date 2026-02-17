#!/bin/bash
# Run vanilla DP-SGD on BraTS at epsilon=8.0 for all 3 seeds
# 
# SIMPLIFIED: No multiprocessing to avoid memory issues
#   - num_workers: 0 (no DataLoader workers)
#   - batch_size: 4 (reduced from 8)
#   - cache_volumes: 50 (moderate cache)

set -e
cd /home/mmolinav/Projects/Seg-DP-KFC/
mkdir -p logs

EPOCHS=10
BATCH=4  # Reduced to avoid OOM
PATCH=64
WORKERS=0  # No multiprocessing
DEVICE=cuda
LR=0.01
EPSILON=8.0
METHOD=dp
CACHE=50

echo "============================================"
echo "BraTS Vanilla DP-SGD (SIMPLIFIED): $(date)"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Method: $METHOD (vanilla DP-SGD)"
echo "  Epsilon: $EPSILON"
echo "  Seeds: 42, 123, 456"
echo "  Learning rate: $LR"
echo "  Batch size: $BATCH"
echo "  Epochs: $EPOCHS"
echo "  Workers: $WORKERS (no multiprocessing)"
echo "  Cache: $CACHE volumes"
echo ""

run_experiment() {
    local seed=$1
    
    echo ""
    echo "============================================"
    echo "Running: method=$METHOD eps=$EPSILON seed=$seed"
    echo "Started: $(date)"
    echo "============================================"
    
    uv run python scripts/train_brats.py \
        --method $METHOD \
        --epochs $EPOCHS \
        --lr $LR \
        --seed $seed \
        --batch_size $BATCH \
        --device $DEVICE \
        --num_workers $WORKERS \
        --patch_size $PATCH \
        --epsilon $EPSILON \
        --cache_volumes $CACHE
    
    echo "Finished: $(date)"
    echo ""
}

# Run all three seeds
for seed in 42 123 456; do
    run_experiment $seed
done

echo ""
echo "============================================"
echo "DP-SGD experiments completed: $(date)"
echo "============================================"

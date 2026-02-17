#!/bin/bash
# BraTS overnight: dp, dp_synthetic (AdaDPS), dp_shampoo_synth × 3 seeds
# patch_size=64, batch_size=16, eps=8.0, 10 epochs
set -euo pipefail

COMMON="--epsilon 8.0 --epochs 10 \
    --data_root data/brats_preprocessed \
    --num_workers 0 --cache_volumes 100 \
    --skip_volume_eval \
    --patch_size 64 --batch_size 16"

for SEED in 42 123 456; do
  for METHOD in dp dp_synthetic dp_shampoo_synth; do
    echo ""
    echo "============================================"
    echo "=== method=$METHOD seed=$SEED ==="
    echo "=== $(date) ==="
    echo "============================================"
    uv run python scripts/train_brats.py \
        --method $METHOD --seed $SEED $COMMON
  done
done

echo ""
echo "=== ALL DONE: $(date) ==="

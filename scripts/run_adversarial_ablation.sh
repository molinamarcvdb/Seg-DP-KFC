#!/bin/bash
# Adversarial preconditioning ablation on FIVES retinal vessel dataset
# Tests whether spatial structure in synthetic data matters for Shampoo preconditioning
# All at eps=8.0, 30 epochs, seed=42, lr=0.01
set -euo pipefail

COMMON="--epsilon 8.0 --epochs 30 --seed 42 --image_size 128 --batch_size 16 --lr 0.01"

echo "=== 1/5: Vanilla DP-SGD (no preconditioning) ==="
echo "=== $(date) ==="
uv run python scripts/train_fundus.py --method dp $COMMON

echo ""
echo "=== 2/5: Shampoo + pink noise + frangi masks (structured both) ==="
echo "=== $(date) ==="
uv run python scripts/train_fundus.py --method dp_shampoo_synth $COMMON --noise_type pink --mask_strategy frangi

echo ""
echo "=== 3/5: Shampoo + white noise + frangi masks (unstructured input) ==="
echo "=== $(date) ==="
uv run python scripts/train_fundus.py --method dp_shampoo_synth $COMMON --noise_type white --mask_strategy frangi

echo ""
echo "=== 4/5: Shampoo + pink noise + random_noise masks (unstructured masks) ==="
echo "=== $(date) ==="
uv run python scripts/train_fundus.py --method dp_shampoo_synth $COMMON --noise_type pink --mask_strategy random_noise

echo ""
echo "=== 5/5: Shampoo + white noise + random_noise masks (unstructured both) ==="
echo "=== $(date) ==="
uv run python scripts/train_fundus.py --method dp_shampoo_synth $COMMON --noise_type white --mask_strategy random_noise

echo ""
echo "=== ALL DONE: $(date) ==="

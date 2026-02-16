#!/bin/bash
set -e

echo "=== LoRA SGD Comparison: dp vs adadps vs shampoo_synth ==="
echo "=== Config: sgd lr=0.005, bs=32, rank=8, alpha=16, refresh_precond=5, eps=8.0 ==="

COMMON="--optimizer sgd --lr 0.005 --epsilon 8.0 --epochs 30 --device cuda --refresh_precond 5 --batch_size 32 --lora_rank 8 --lora_alpha 16"

echo ""
echo "=== [1/3] lora_dp (vanilla DP-SGD) ==="
uv run python scripts/train_lora_dp.py --method lora_dp $COMMON

echo ""
echo "=== [2/3] lora_dp_adadps (AdaDPS synthetic) ==="
uv run python scripts/train_lora_dp.py --method lora_dp_adadps $COMMON

echo ""
echo "=== [3/3] lora_dp_shampoo_synth (Shampoo synthetic) ==="
uv run python scripts/train_lora_dp.py --method lora_dp_shampoo_synth $COMMON

echo ""
echo "=== ALL DONE ==="

#!/bin/bash
set -e

echo "=== LoRA SGD across 3 seeds: dp vs adadps vs shampoo_synth, eps=8.0 ==="
echo "=== Config: sgd lr=0.005, bs=32, rank=8, alpha=16, refresh_precond=5 ==="

COMMON="--optimizer sgd --lr 0.005 --epsilon 8.0 --epochs 30 --device cuda --refresh_precond 5 --batch_size 32 --lora_rank 8 --lora_alpha 16"

for SEED in 42 123 456; do
  echo ""
  echo "========== SEED=$SEED =========="

  echo "=== lora_dp seed=$SEED ==="
  uv run python scripts/train_lora_dp.py --method lora_dp $COMMON --seed $SEED

  echo "=== lora_dp_adadps seed=$SEED ==="
  uv run python scripts/train_lora_dp.py --method lora_dp_adadps $COMMON --seed $SEED

  echo "=== lora_dp_shampoo_synth seed=$SEED ==="
  uv run python scripts/train_lora_dp.py --method lora_dp_shampoo_synth $COMMON --seed $SEED
done

echo ""
echo "=== ALL DONE ==="

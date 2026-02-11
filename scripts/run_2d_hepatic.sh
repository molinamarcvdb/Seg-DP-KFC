#!/bin/bash
# Run Kvasir + Fundus + Hepatic experiments for MICCAI 2026
# Phase 1: Main table (eps=8.0, 6 methods, 3 seeds) — 54 runs
# Phase 2: Epsilon sweep (eps=1,2,4,8, 4 DP methods, 3 seeds) — 153 runs
# Total: 207 runs
#
# Kvasir:  30 epochs, batch=16, ~4 min/run  → ~5h
# Fundus:  30 epochs, batch=16, ~5 min/run  → ~6h
# Hepatic: 10 epochs, batch=16, patch=64, ~45 min/run → ~52h
#
# Estimated total: ~63 hours (~2.6 days)
#
# Usage: nohup bash scripts/run_2d_hepatic.sh > logs/2d_hepatic.log 2>&1 &

set -e
cd /home/mmolinav/Projects/SEG_DP_KFC
mkdir -p logs

echo "============================================"
echo "Starting 2D + Hepatic experiments: $(date)"
echo "Datasets: kvasir, fundus, hepatic"
echo "============================================"

# Phase 1: Main table
echo ""
echo ">>> PHASE 1: Main table (eps=8.0, 6 methods × 3 seeds = 54 runs)"
echo ""
uv run python scripts/run_all_experiments.py --phase main --dataset kvasir fundus hepatic

# Phase 2: Epsilon sweep
echo ""
echo ">>> PHASE 2: Epsilon sweep (4 eps × 4 DP methods × 3 seeds = 153 runs)"
echo ""
uv run python scripts/run_all_experiments.py --phase sweep --dataset kvasir fundus hepatic

echo ""
echo "============================================"
echo "All 2D + Hepatic experiments finished: $(date)"
echo "============================================"
echo "Run: uv run python scripts/analyze_results.py"

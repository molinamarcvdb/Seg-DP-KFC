#!/bin/bash
# Run BraTS experiments standalone (for separate machine)
# Phase 1: Main table (eps=8.0, 6 methods, 3 seeds) — 18 runs
# Phase 2: Epsilon sweep (eps=1,2,4,8, 4 DP methods, 3 seeds) — 51 runs
# Total: 69 runs
#
# BraTS: 10 epochs, batch=8, patch=64, ~3.5h/run avg
#
# Estimated total: ~240 hours (~10 days)
#
# Usage: nohup bash scripts/run_brats_standalone.sh > logs/brats_standalone.log 2>&1 &

set -e
cd /home/mmolinav/Projects/Seg-DP-KFC/
mkdir -p logs

echo "============================================"
echo "Starting BraTS experiments: $(date)"
echo "============================================"

# Phase 1: Main table
echo ""
echo ">>> PHASE 1: Main table (eps=8.0, 6 methods × 3 seeds = 18 runs)"
echo ""
uv run python scripts/run_all_experiments.py --phase main --dataset brats

# Phase 2: Epsilon sweep
echo ""
echo ">>> PHASE 2: Epsilon sweep (4 eps × 4 DP methods × 3 seeds = 51 runs)"
echo ""
uv run python scripts/run_all_experiments.py --phase sweep --dataset brats

echo ""
echo "============================================"
echo "All BraTS experiments finished: $(date)"
echo "============================================"
echo "Run: uv run python scripts/analyze_results.py"

#!/bin/bash
# Run full BraTS + Hepatic experiments for MICCAI 2026
# Phase 1: Main table (eps=8.0, 3 seeds, all methods)
# Phase 2: Epsilon sweep (eps=1,2,4,8, 3 seeds, DP methods)
#
# BraTS:   batch=8, patch=64, 10 epochs  (~30min/run)
# Hepatic: batch=16, patch=64, 10 epochs (~15min/run)
#
# Estimated total: ~65 hours (~2.7 days)
#
# Usage: nohup uv run bash scripts/run_3d_experiments.sh > logs/3d_experiments.log 2>&1 &

set -e
cd /home/mmolinav/Projects/SEG_DP_KFC

echo "============================================"
echo "Starting 3D experiments: $(date)"
echo "============================================"

# Phase 1: Main table for BraTS + Hepatic
echo ""
echo ">>> PHASE 1: Main table (BraTS + Hepatic)"
echo ">>> Expected: 36 runs"
echo ""
uv run python scripts/run_all_experiments.py --phase main --dataset brats hepatic

# Phase 2: Epsilon sweep for BraTS + Hepatic
echo ""
echo ">>> PHASE 2: Epsilon sweep (BraTS + Hepatic)"
echo ">>> Expected: 102 runs"
echo ""
uv run python scripts/run_all_experiments.py --phase sweep --dataset brats hepatic

echo ""
echo "============================================"
echo "All 3D experiments finished: $(date)"
echo "============================================"
echo "Run: uv run python scripts/analyze_results.py"

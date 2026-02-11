#!/bin/bash
# BraTS overnight experiment suite
# patch_size=128, 25% data, 5 epochs, eps={1,2,4,8}, 1 seed
# Methods: baseline, dp, dp_synthetic (AdaDPS), dp_shampoo (oracle), dp_shampoo_synth
# 17 runs total, estimated ~13 hours

set -e
cd /home/mmolinav/Projects/SEG_DP_KFC

LOGDIR="logs/brats_overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

COMMON="--patch_size 128 --subset_fraction 0.25 --num_workers 4 --eval_volumes 10"
EPSILONS="1.0 2.0 4.0 8.0"
RUN=1
TOTAL=17

echo "=== BraTS overnight run started at $(date) ===" | tee "$LOGDIR/summary.log"
echo "Logs in: $LOGDIR" | tee -a "$LOGDIR/summary.log"
echo "Plan: 17 runs (1 baseline + 4x4 DP methods), ~13 hours" | tee -a "$LOGDIR/summary.log"

# ---------- BASELINE (no DP, bs=4 since no Opacus overhead) ----------
echo "" | tee -a "$LOGDIR/summary.log"
echo ">>> [$RUN/$TOTAL] baseline, 5 epochs (started $(date +%H:%M))" | tee -a "$LOGDIR/summary.log"
uv run python scripts/train_brats.py --method baseline --epochs 5 --batch_size 4 $COMMON \
    > "$LOGDIR/baseline.log" 2>&1
tail -8 "$LOGDIR/baseline.log" | tee -a "$LOGDIR/summary.log"
RUN=$((RUN + 1))

# ---------- DP (vanilla) ----------
for EPS in $EPSILONS; do
    echo "" | tee -a "$LOGDIR/summary.log"
    echo ">>> [$RUN/$TOTAL] dp, eps=$EPS, 5 epochs (started $(date +%H:%M))" | tee -a "$LOGDIR/summary.log"
    uv run python scripts/train_brats.py --method dp --epochs 5 --batch_size 1 --epsilon $EPS $COMMON \
        > "$LOGDIR/dp_eps${EPS}.log" 2>&1
    tail -8 "$LOGDIR/dp_eps${EPS}.log" | tee -a "$LOGDIR/summary.log"
    RUN=$((RUN + 1))
done

# ---------- DP + AdaDPS (synthetic diagonal preconditioning) ----------
for EPS in $EPSILONS; do
    echo "" | tee -a "$LOGDIR/summary.log"
    echo ">>> [$RUN/$TOTAL] dp_synthetic (AdaDPS), eps=$EPS, 5 epochs (started $(date +%H:%M))" | tee -a "$LOGDIR/summary.log"
    uv run python scripts/train_brats.py --method dp_synthetic --epochs 5 --batch_size 1 --epsilon $EPS $COMMON \
        > "$LOGDIR/dp_synthetic_eps${EPS}.log" 2>&1
    tail -8 "$LOGDIR/dp_synthetic_eps${EPS}.log" | tee -a "$LOGDIR/summary.log"
    RUN=$((RUN + 1))
done

# ---------- DP + Shampoo (oracle = public train data) ----------
for EPS in $EPSILONS; do
    echo "" | tee -a "$LOGDIR/summary.log"
    echo ">>> [$RUN/$TOTAL] dp_shampoo (oracle), eps=$EPS, 5 epochs (started $(date +%H:%M))" | tee -a "$LOGDIR/summary.log"
    uv run python scripts/train_brats.py --method dp_shampoo --epochs 5 --batch_size 1 --epsilon $EPS $COMMON \
        > "$LOGDIR/dp_shampoo_eps${EPS}.log" 2>&1
    tail -8 "$LOGDIR/dp_shampoo_eps${EPS}.log" | tee -a "$LOGDIR/summary.log"
    RUN=$((RUN + 1))
done

# ---------- DP + Shampoo (synthetic) ----------
for EPS in $EPSILONS; do
    echo "" | tee -a "$LOGDIR/summary.log"
    echo ">>> [$RUN/$TOTAL] dp_shampoo_synth, eps=$EPS, 5 epochs (started $(date +%H:%M))" | tee -a "$LOGDIR/summary.log"
    uv run python scripts/train_brats.py --method dp_shampoo_synth --epochs 5 --batch_size 1 --epsilon $EPS $COMMON \
        > "$LOGDIR/dp_shampoo_synth_eps${EPS}.log" 2>&1
    tail -8 "$LOGDIR/dp_shampoo_synth_eps${EPS}.log" | tee -a "$LOGDIR/summary.log"
    RUN=$((RUN + 1))
done

echo "" | tee -a "$LOGDIR/summary.log"
echo "=== All done at $(date) ===" | tee -a "$LOGDIR/summary.log"

# ---------- Collect results into a table ----------
echo "" | tee -a "$LOGDIR/summary.log"
echo "=== RESULTS TABLE ===" | tee -a "$LOGDIR/summary.log"
uv run python -c "
import json, glob
from pathlib import Path

results = []
for f in sorted(glob.glob('outputs/brats_*_*/results.json')):
    with open(f) as fp:
        r = json.load(fp)
    results.append(r)

# Deduplicate: keep latest per (method, epsilon)
seen = {}
for r in results:
    key = (r['method'], r.get('epsilon'))
    seen[key] = r

print(f\"{'Method':<25} {'Eps':>5} {'Patch Dice':>11} {'Vol Dice':>10}\")
print('-' * 55)
for (method, eps), r in sorted(seen.items(), key=lambda x: (x[0][0], x[0][1] or 0)):
    eps_str = f\"{eps}\" if eps else 'N/A'
    print(f\"{method:<25} {eps_str:>5} {r['best_val_dice']:>11.4f} {r.get('full_volume_dice', 0):>10.4f}\")
" 2>&1 | tee -a "$LOGDIR/summary.log"

#!/usr/bin/env python3
"""
Run ALL experiments for MICCAI 2026 paper.

This script orchestrates the complete experiment matrix:
  - 4 datasets: Kvasir-SEG, Fundus (2D seg), BraTS (3D seg), Hepatic (3D seg)
  - 6 methods: baseline, dp, dp_synthetic (AdaDPS), dp_public (AdaDPS oracle), dp_shampoo (Shampoo oracle), dp_shampoo_synth (Shampoo)
  - 4 epsilon values: 1.0, 2.0, 4.0, 8.0
  - 3 seeds: 42, 123, 456

Phases:
  1. tune  - Find best LR per method/dataset (5 epochs each, 5 LRs)
  2. main  - Fill the main results table (all methods x datasets, eps=8.0, 3 seeds)
  3. sweep - Epsilon sweep for the paper figure (eps=1,2,4,8 x 3 methods x 3 seeds)
  4. all   - Run everything

Usage:
    # Run everything (tune + main + sweep)
    python scripts/run_all_experiments.py --phase all

    # Just fill the main table
    python scripts/run_all_experiments.py --phase main

    # Epsilon sweep only (assumes LR already tuned)
    python scripts/run_all_experiments.py --phase sweep --dataset kvasir

    # Dry run (print what would be done)
    python scripts/run_all_experiments.py --phase all --dry_run
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import subprocess
import time
from datetime import datetime


# ============================================================================
# Experiment Configuration
# ============================================================================

DATASETS = ["kvasir", "fundus", "brats", "hepatic"]
METHODS = ["baseline", "dp", "dp_synthetic", "dp_public", "dp_shampoo", "dp_shampoo_synth"]
SEEDS = [42, 123, 456]
EPSILONS = [1.0, 2.0, 4.0, 8.0]
DEFAULT_EPSILON = 8.0

# Tuned LRs from prior experiments (updated after tuning phase)
# Format: {(dataset, method): lr}
TUNED_LRS = {
    # PathMNIST - classification (from prior runs)
    ("pathmnist", "baseline"): 0.001,
    ("pathmnist", "dp"): 0.05,
    ("pathmnist", "dp_synthetic"): 0.01,
    ("pathmnist", "dp_public"): 0.01,
    ("pathmnist", "dp_shampoo"): 0.01,
    ("pathmnist", "dp_shampoo_synth"): 0.01,
    # Kvasir - segmentation (from MICCAI runs)
    ("kvasir", "baseline"): 0.05,
    ("kvasir", "dp"): 0.1,
    ("kvasir", "dp_synthetic"): 0.05,
    ("kvasir", "dp_public"): 0.05,
    ("kvasir", "dp_shampoo"): 0.1,
    ("kvasir", "dp_shampoo_synth"): 0.1,
    # Pet - segmentation (from prior runs)
    ("pet", "baseline"): 0.01,
    ("pet", "dp"): 0.01,
    ("pet", "dp_synthetic"): 0.01,
    ("pet", "dp_public"): 0.01,
    ("pet", "dp_shampoo"): 0.05,
    ("pet", "dp_shampoo_synth"): 0.05,
    # Fundus - segmentation (LR TBD after tuning)
    ("fundus", "baseline"): 0.05,
    ("fundus", "dp"): 0.1,
    ("fundus", "dp_synthetic"): 0.05,
    ("fundus", "dp_public"): 0.05,
    ("fundus", "dp_shampoo"): 0.1,
    ("fundus", "dp_shampoo_synth"): 0.1,
    # BraTS - 3D segmentation (LR TBD after tuning)
    ("brats", "baseline"): 0.001,
    ("brats", "dp"): 0.01,
    ("brats", "dp_synthetic"): 0.01,
    ("brats", "dp_public"): 0.01,
    ("brats", "dp_shampoo"): 0.01,
    ("brats", "dp_shampoo_synth"): 0.01,
    # Hepatic Vessel - 3D segmentation (LR TBD after tuning)
    ("hepatic", "baseline"): 0.001,
    ("hepatic", "dp"): 0.01,
    ("hepatic", "dp_synthetic"): 0.01,
    ("hepatic", "dp_public"): 0.01,
    ("hepatic", "dp_shampoo"): 0.01,
    ("hepatic", "dp_shampoo_synth"): 0.01,
    # LoRA methods (Kvasir only for now, LR TBD after tuning)
    ("kvasir", "lora_baseline"): 1e-4,
    ("kvasir", "lora_dp"): 1e-4,
    ("kvasir", "lora_dp_shampoo"): 1e-4,
    ("kvasir", "lora_dp_shampoo_synth"): 1e-4,
    ("kvasir", "lora_dp_adadps"): 1e-4,
    ("kvasir", "lora_dp_ffa"): 1e-4,
    ("kvasir", "lora_dp_ffa_shampoo"): 1e-4,
}

LORA_METHODS = [
    "lora_baseline", "lora_dp", "lora_dp_shampoo", "lora_dp_shampoo_synth",
    "lora_dp_adadps", "lora_dp_ffa", "lora_dp_ffa_shampoo",
]

# Dataset-specific settings
DATASET_CONFIGS = {
    "pathmnist": {"epochs": 30, "tune_epochs": 5, "batch_size": 128, "script": "train_medmnist.py",
                  "extra_args": ["--optimizer", "sgd", "--model", "large_cnn"]},
    "kvasir":    {"epochs": 30, "tune_epochs": 5, "batch_size": 16,  "script": "train_kvasir.py",
                  "extra_args": []},
    "pet":       {"epochs": 30, "tune_epochs": 5, "batch_size": 32,  "script": "train_pet_segmentation.py",
                  "extra_args": []},
    "fundus":    {"epochs": 30, "tune_epochs": 5, "batch_size": 16,  "script": "train_fundus.py",
                  "extra_args": []},
    "brats":     {"epochs": 10, "tune_epochs": 3, "batch_size": 8,   "script": "train_brats.py",
                  "extra_args": ["--num_workers", "4", "--patch_size", "64"]},
    "hepatic":   {"epochs": 10, "tune_epochs": 3, "batch_size": 16,  "script": "train_hepatic.py",
                  "extra_args": ["--num_workers", "4", "--patch_size", "64"]},
}


def run_cmd(cmd, dry_run=False):
    """Run a command and return success/failure."""
    print(f"\n{'─'*60}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'─'*60}")

    if dry_run:
        print("  [DRY RUN] Skipped")
        return True, 0

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)
        elapsed = time.time() - start
        success = result.returncode == 0
        if not success:
            print(f"  [FAILED] Return code: {result.returncode}")
        else:
            print(f"  [OK] Completed in {elapsed:.0f}s")
        return success, elapsed
    except subprocess.TimeoutExpired:
        print("  [TIMEOUT] Exceeded 2 hours")
        return False, 7200
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False, 0


# ============================================================================
# Phase 1: LR Tuning
# ============================================================================

def phase_tune(datasets=None, methods=None, device="cuda", dry_run=False):
    """Tune LR for each dataset/method."""
    datasets = datasets or DATASETS
    methods = methods or METHODS

    print("\n" + "="*60)
    print("PHASE 1: LEARNING RATE TUNING")
    print("="*60)

    results = {}
    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        for method in methods:
            cmd = [
                sys.executable, "scripts/tune_and_train.py",
                "--dataset", ds,
                "--method", method,
                "--phase", "tune",
                "--tune_epochs", str(cfg["tune_epochs"]),
                "--epsilon", str(DEFAULT_EPSILON),
                "--device", device,
            ]
            success, elapsed = run_cmd(cmd, dry_run)
            results[(ds, method)] = {"success": success, "elapsed": elapsed}

    return results


# ============================================================================
# Phase 2: Main Table (all methods x datasets x seeds, eps=8.0)
# ============================================================================

def phase_main(datasets=None, methods=None, device="cuda", dry_run=False):
    """Run main experiment table with 3 seeds."""
    datasets = datasets or DATASETS
    methods = methods or METHODS

    print("\n" + "="*60)
    print("PHASE 2: MAIN RESULTS TABLE")
    print("="*60)

    results = []
    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        for method in methods:
            lr = TUNED_LRS.get((ds, method), 0.01)
            for seed in SEEDS:
                if method == "baseline" and seed != SEEDS[0]:
                    # Baseline with different seeds still runs (for error bars)
                    pass  # run all seeds

                cmd = [
                    sys.executable, f"scripts/{cfg['script']}",
                    "--method", method,
                    "--epochs", str(cfg["epochs"]),
                    "--lr", str(lr),
                    "--seed", str(seed),
                    "--batch_size", str(cfg["batch_size"]),
                    "--device", device,
                ] + cfg.get("extra_args", [])
                if method != "baseline":
                    cmd += ["--epsilon", str(DEFAULT_EPSILON)]

                success, elapsed = run_cmd(cmd, dry_run)
                results.append({
                    "dataset": ds, "method": method, "seed": seed,
                    "epsilon": DEFAULT_EPSILON, "lr": lr,
                    "success": success, "elapsed": elapsed,
                })

    return results


# ============================================================================
# Phase 3: Epsilon Sweep
# ============================================================================

def phase_sweep(datasets=None, device="cuda", dry_run=False):
    """Epsilon sweep on selected datasets."""
    datasets = datasets or ["kvasir"]  # default to kvasir for sweep
    sweep_methods = ["dp", "dp_synthetic", "dp_shampoo", "dp_shampoo_synth"]

    print("\n" + "="*60)
    print("PHASE 3: EPSILON SWEEP")
    print("="*60)

    results = []
    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        # Run baseline once (no epsilon dependence)
        lr = TUNED_LRS.get((ds, "baseline"), 0.01)
        for seed in SEEDS:
            cmd = [
                sys.executable, f"scripts/{cfg['script']}",
                "--method", "baseline",
                "--epochs", str(cfg["epochs"]),
                "--lr", str(lr),
                "--seed", str(seed),
                "--batch_size", str(cfg["batch_size"]),
                "--device", device,
            ] + cfg.get("extra_args", [])
            success, elapsed = run_cmd(cmd, dry_run)
            results.append({
                "dataset": ds, "method": "baseline", "seed": seed,
                "epsilon": None, "success": success,
            })

        # DP methods x epsilons x seeds
        for method in sweep_methods:
            lr = TUNED_LRS.get((ds, method), 0.01)
            for eps in EPSILONS:
                for seed in SEEDS:
                    cmd = [
                        sys.executable, f"scripts/{cfg['script']}",
                        "--method", method,
                        "--epochs", str(cfg["epochs"]),
                        "--lr", str(lr),
                        "--seed", str(seed),
                        "--batch_size", str(cfg["batch_size"]),
                        "--epsilon", str(eps),
                        "--device", device,
                    ] + cfg.get("extra_args", [])
                    success, elapsed = run_cmd(cmd, dry_run)
                    results.append({
                        "dataset": ds, "method": method, "seed": seed,
                        "epsilon": eps, "success": success,
                    })

    return results


# ============================================================================
# Phase 4: LoRA Experiments
# ============================================================================

def phase_lora(device="cuda", dry_run=False):
    """Run DP-LoRA finetuning experiments (Experiment 6)."""
    print("\n" + "="*60)
    print("PHASE 4: DP-LORA FINETUNING")
    print("="*60)

    results = []
    lora_dp_methods = [m for m in LORA_METHODS if m != "lora_baseline"]

    # Baseline (no epsilon dependence)
    lr = TUNED_LRS.get(("kvasir", "lora_baseline"), 1e-4)
    for seed in SEEDS:
        cmd = [
            sys.executable, "scripts/train_lora_dp.py",
            "--method", "lora_baseline",
            "--epochs", "30",
            "--lr", str(lr),
            "--seed", str(seed),
            "--batch_size", "8",
            "--device", device,
        ]
        success, elapsed = run_cmd(cmd, dry_run)
        results.append({
            "dataset": "kvasir", "method": "lora_baseline",
            "seed": seed, "epsilon": None, "success": success,
        })

    # DP methods x epsilons x seeds
    for method in lora_dp_methods:
        lr = TUNED_LRS.get(("kvasir", method), 1e-4)
        for eps in EPSILONS:
            for seed in SEEDS:
                cmd = [
                    sys.executable, "scripts/train_lora_dp.py",
                    "--method", method,
                    "--epochs", "30",
                    "--lr", str(lr),
                    "--seed", str(seed),
                    "--batch_size", "8",
                    "--epsilon", str(eps),
                    "--device", device,
                ]
                success, elapsed = run_cmd(cmd, dry_run)
                results.append({
                    "dataset": "kvasir", "method": method,
                    "seed": seed, "epsilon": eps, "success": success,
                })

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run all MICCAI experiments")
    parser.add_argument("--phase", type=str, default="main",
                        choices=["tune", "main", "sweep", "lora", "all"])
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        choices=DATASETS,
                        help="Datasets to run (default: all)")
    parser.add_argument("--method", type=str, nargs="+", default=None,
                        choices=METHODS,
                        help="Methods to run (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    start_time = time.time()
    all_results = {}

    if args.phase in ["tune", "all"]:
        all_results["tune"] = phase_tune(args.dataset, args.method, args.device, args.dry_run)

    if args.phase in ["main", "all"]:
        all_results["main"] = phase_main(args.dataset, args.method, args.device, args.dry_run)

    if args.phase in ["sweep", "all"]:
        all_results["sweep"] = phase_sweep(args.dataset, args.device, args.dry_run)

    if args.phase in ["lora", "all"]:
        all_results["lora"] = phase_lora(args.device, args.dry_run)

    total_time = time.time() - start_time

    # Save run log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "phase": args.phase,
        "datasets": args.dataset or DATASETS,
        "total_time_seconds": total_time,
        "results": {k: str(v) for k, v in all_results.items()},
    }
    log_path = Path("outputs") / f"run_log_{timestamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL DONE in {total_time/60:.1f} minutes")
    print(f"Run log: {log_path}")
    print(f"{'='*60}")

    # Print what to do next
    print(f"\nNext steps:")
    print(f"  1. Review results:  python scripts/analyze_results.py")
    print(f"  2. Update TUNED_LRS in this script with tuning results")
    print(f"  3. Run missing experiments if any failed")


if __name__ == "__main__":
    main()

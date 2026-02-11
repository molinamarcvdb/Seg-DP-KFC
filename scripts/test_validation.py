#!/usr/bin/env python3
"""
Quick validation test for all preconditioner methods.

Runs all methods on Kvasir-SEG (20 epochs, eps=8.0) to validate
the approach before committing to full experiment runs.

Methods tested:
  1. baseline       - No DP (upper bound)
  2. dp             - Standard DP-SGD (lower bound)
  3. dp_adadps      - DP-SGD + AdaDPS diagonal preconditioner (public data)
  4. dp_kfac        - DP-SGD + K-FAC preconditioner (public data)
  5. dp_adadps_synth - DP-SGD + AdaDPS from synthetic data
  6. dp_kfac_synth  - DP-SGD + K-FAC from synthetic data

Success criteria:
  - dp_kfac > dp (K-FAC helps DP-SGD)
  - dp_kfac ≈ dp_adadps or better (K-FAC >= diagonal)
  - dp_*_synth ≈ dp_*_public (synthetic is comparable)

Usage:
    python scripts/test_validation.py
    python scripts/test_validation.py --epochs 5 --device cpu  # quick smoke test
    python scripts/test_validation.py --dataset pathmnist      # classification
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import GradSampleModule

from src.data import KvasirSegDataset, MedMNISTDataset
from src.models import create_model, create_classifier
from src.training import (
    DPTrainer,
    NonDPTrainer,
    DPClassificationTrainer,
    NonDPClassificationTrainer,
    AdaDPSPreconditioner,
    KFACPreconditioner,
    ShampooPreconditioner,
    MomentumPreconditioner,
    DiceBCELoss,
)


def run_kvasir_experiment(method, epochs, epsilon, lr, device, batch_size=16,
                          image_size=128, damping=0.1, kfac_damping=1e-3):
    """Run a single Kvasir experiment and return metrics."""
    torch.manual_seed(42)

    # Data
    train_ds = KvasirSegDataset(split="train", image_size=image_size, augment=True)
    val_ds = KvasirSegDataset(split="val", image_size=image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = create_model("unet", in_channels=3, out_channels=1, features=[32, 64, 128])
    loss_fn = DiceBCELoss()

    def make_fresh():
        m = create_model("unet", in_channels=3, out_channels=1, features=[32, 64, 128])
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        return m, opt, sch

    preconditioner = None
    start = time.time()

    if method == "baseline":
        model, optimizer, scheduler = make_fresh()
        trainer = NonDPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                               epochs, device, scheduler=scheduler)

    elif method == "dp":
        model, optimizer, scheduler = make_fresh()
        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           epsilon, 1e-5, 1.0, epochs, device, preconditioner=None,
                           scheduler=scheduler)

    elif method == "dp_adadps":
        # AdaDPS from public (train) data
        preconditioner = AdaDPSPreconditioner(damping=damping, device=device)
        wrapped = GradSampleModule(model).to(device)
        preconditioner.estimate_from_loader(wrapped, train_loader, loss_fn,
                                            num_steps=min(30, len(train_loader)),
                                            task="segmentation")
        model, optimizer, scheduler = make_fresh()
        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           epsilon, 1e-5, 1.0, epochs, device,
                           preconditioner=preconditioner, scheduler=scheduler)

    elif method == "dp_kfac":
        # K-FAC from public (train) data
        preconditioner = KFACPreconditioner(damping=kfac_damping, device=device)
        wrapped = GradSampleModule(model).to(device)
        preconditioner.estimate_from_loader(wrapped, train_loader, loss_fn,
                                            num_steps=min(100, len(train_loader)),
                                            task="segmentation")
        model, optimizer, scheduler = make_fresh()
        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           epsilon, 1e-5, 1.0, epochs, device,
                           preconditioner=preconditioner, scheduler=scheduler)

    elif method == "dp_adadps_synth":
        # AdaDPS from synthetic data
        preconditioner = AdaDPSPreconditioner(damping=damping, device=device)
        wrapped = GradSampleModule(model).to(device)
        preconditioner.estimate_from_synthetic(wrapped, loss_fn, num_samples=500,
                                               batch_size=batch_size, image_size=image_size,
                                               in_channels=3, num_classes=1,
                                               task="segmentation", mask_type="blob")
        model, optimizer, scheduler = make_fresh()
        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           epsilon, 1e-5, 1.0, epochs, device,
                           preconditioner=preconditioner, scheduler=scheduler)

    elif method == "dp_kfac_synth":
        # K-FAC from synthetic data
        preconditioner = KFACPreconditioner(damping=kfac_damping, device=device)
        wrapped = GradSampleModule(model).to(device)
        preconditioner.estimate_from_synthetic(wrapped, loss_fn, num_samples=1000,
                                               batch_size=batch_size, image_size=image_size,
                                               in_channels=3, num_classes=1,
                                               task="segmentation", mask_type="blob")
        model, optimizer, scheduler = make_fresh()
        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           epsilon, 1e-5, 1.0, epochs, device,
                           preconditioner=preconditioner, scheduler=scheduler)

    elif method == "dp_shampoo":
        # Shampoo from public (train) data
        preconditioner = ShampooPreconditioner(damping=1e-4, device=device)
        wrapped = GradSampleModule(model).to(device)
        preconditioner.estimate_from_loader(wrapped, train_loader, loss_fn,
                                            num_steps=min(100, len(train_loader)),
                                            task="segmentation")
        model, optimizer, scheduler = make_fresh()
        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           epsilon, 1e-5, 1.0, epochs, device,
                           preconditioner=preconditioner, scheduler=scheduler)

    elif method == "dp_shampoo_synth":
        # Shampoo from synthetic data
        preconditioner = ShampooPreconditioner(damping=1e-4, device=device)
        wrapped = GradSampleModule(model).to(device)
        preconditioner.estimate_from_synthetic(wrapped, loss_fn, num_samples=1000,
                                               batch_size=batch_size, image_size=image_size,
                                               in_channels=3, num_classes=1,
                                               task="segmentation", mask_type="blob")
        model, optimizer, scheduler = make_fresh()
        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           epsilon, 1e-5, 1.0, epochs, device,
                           preconditioner=preconditioner, scheduler=scheduler)

    elif method == "dp_momentum":
        # Momentum preconditioner from synthetic (with fixed defaults)
        preconditioner = MomentumPreconditioner(damping=damping, beta2=0.9, device=device)
        wrapped = GradSampleModule(model).to(device)
        preconditioner.estimate_from_synthetic(wrapped, loss_fn, num_samples=500,
                                               batch_size=batch_size, image_size=image_size,
                                               in_channels=3, num_classes=1,
                                               task="segmentation", mask_type="blob")
        model, optimizer, scheduler = make_fresh()
        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           epsilon, 1e-5, 1.0, epochs, device,
                           preconditioner=preconditioner, scheduler=scheduler)

    history = trainer.fit()
    elapsed = time.time() - start

    return {
        "best_dice": max(history["val_dice"]),
        "final_dice": history["val_dice"][-1],
        "history": history,
        "elapsed": elapsed,
    }


def run_pathmnist_experiment(method, epochs, epsilon, lr, device, batch_size=128,
                             damping=0.1, kfac_damping=1e-3):
    """Run a single PathMNIST experiment and return metrics."""
    torch.manual_seed(42)

    train_ds = MedMNISTDataset("pathmnist", split="train", download=True)
    val_ds = MedMNISTDataset("pathmnist", split="val", download=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = create_classifier("large_cnn", in_channels=3, num_classes=9)
    loss_fn = nn.CrossEntropyLoss()

    def make_fresh():
        m = create_classifier("large_cnn", in_channels=3, num_classes=9)
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        return m, opt, sch

    preconditioner = None
    start = time.time()

    if method == "baseline":
        model, optimizer, scheduler = make_fresh()
        trainer = NonDPClassificationTrainer(model, train_loader, val_loader, optimizer,
                                             loss_fn, epochs, device, scheduler=scheduler)

    elif method == "dp":
        model, optimizer, scheduler = make_fresh()
        trainer = DPClassificationTrainer(model, train_loader, val_loader, optimizer,
                                          loss_fn, epsilon, 1e-5, 1.0, epochs, device,
                                          preconditioner=None, scheduler=scheduler)

    elif method == "dp_adadps":
        preconditioner = AdaDPSPreconditioner(damping=damping, device=device)
        wrapped = GradSampleModule(model).to(device)
        preconditioner.estimate_from_loader(wrapped, train_loader, loss_fn,
                                            num_steps=min(30, len(train_loader)),
                                            task="classification")
        model, optimizer, scheduler = make_fresh()
        trainer = DPClassificationTrainer(model, train_loader, val_loader, optimizer,
                                          loss_fn, epsilon, 1e-5, 1.0, epochs, device,
                                          preconditioner=preconditioner, scheduler=scheduler)

    elif method == "dp_kfac":
        preconditioner = KFACPreconditioner(damping=kfac_damping, device=device)
        wrapped = GradSampleModule(model).to(device)
        preconditioner.estimate_from_loader(wrapped, train_loader, loss_fn,
                                            num_steps=min(100, len(train_loader)),
                                            task="classification")
        model, optimizer, scheduler = make_fresh()
        trainer = DPClassificationTrainer(model, train_loader, val_loader, optimizer,
                                          loss_fn, epsilon, 1e-5, 1.0, epochs, device,
                                          preconditioner=preconditioner, scheduler=scheduler)

    elif method == "dp_kfac_synth":
        preconditioner = KFACPreconditioner(damping=kfac_damping, device=device)
        wrapped = GradSampleModule(model).to(device)
        preconditioner.estimate_from_synthetic(wrapped, loss_fn, num_samples=1000,
                                               batch_size=batch_size, image_size=28,
                                               in_channels=3, num_classes=9,
                                               task="classification")
        model, optimizer, scheduler = make_fresh()
        trainer = DPClassificationTrainer(model, train_loader, val_loader, optimizer,
                                          loss_fn, epsilon, 1e-5, 1.0, epochs, device,
                                          preconditioner=preconditioner, scheduler=scheduler)

    history = trainer.fit()
    elapsed = time.time() - start

    return {
        "best_acc": max(history["val_acc"]),
        "final_acc": history["val_acc"][-1],
        "history": history,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Validation test for preconditioners")
    parser.add_argument("--dataset", type=str, default="kvasir",
                        choices=["kvasir", "pathmnist"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="Specific methods to test")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    if args.dataset == "kvasir":
        all_methods = ["baseline", "dp", "dp_adadps", "dp_kfac",
                       "dp_shampoo", "dp_shampoo_synth",
                       "dp_adadps_synth", "dp_kfac_synth", "dp_momentum"]
        lr = args.lr or 0.05
        run_fn = run_kvasir_experiment
        metric_name = "best_dice"
    else:
        all_methods = ["baseline", "dp", "dp_adadps", "dp_kfac", "dp_kfac_synth"]
        lr = args.lr or 0.01
        run_fn = run_pathmnist_experiment
        metric_name = "best_acc"

    methods = args.methods or all_methods

    print(f"{'='*70}")
    print(f"VALIDATION TEST: {args.dataset}, {args.epochs} epochs, eps={args.epsilon}")
    print(f"Methods: {', '.join(methods)}")
    print(f"{'='*70}")

    results = {}
    for method in methods:
        print(f"\n{'─'*70}")
        print(f"Running: {method}")
        print(f"{'─'*70}")

        try:
            r = run_fn(method, args.epochs, args.epsilon, lr, args.device)
            results[method] = r
            print(f"\n  >> {metric_name}: {r[metric_name]:.4f} ({r['elapsed']:.0f}s)")
        except Exception as e:
            print(f"\n  >> FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[method] = {"error": str(e)}

    # Summary table
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY ({args.dataset}, eps={args.epsilon}, {args.epochs} epochs)")
    print(f"{'='*70}")
    print(f"{'Method':<20s}  {metric_name:<12s}  {'Time':>8s}")
    print(f"{'─'*45}")

    baseline_val = None
    dp_val = None
    for method in methods:
        r = results.get(method, {})
        if "error" in r:
            print(f"{method:<20s}  {'FAILED':<12s}")
            continue

        val = r.get(metric_name, 0)
        elapsed = r.get("elapsed", 0)
        marker = ""

        if method == "baseline":
            baseline_val = val
        elif method == "dp":
            dp_val = val

        # Show improvement markers
        if dp_val is not None and method not in ["baseline", "dp"]:
            diff = val - dp_val
            if diff > 0:
                marker = f"  (+{diff:.4f} vs dp)"
            else:
                marker = f"  ({diff:.4f} vs dp)"

        print(f"{method:<20s}  {val:<12.4f}  {elapsed:>7.0f}s{marker}")

    if baseline_val and dp_val:
        print(f"\n  DP gap from baseline: {dp_val - baseline_val:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"validation_{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_results = {}
    for method, r in results.items():
        save_results[method] = {k: v for k, v in r.items() if k != "history"}

    with open(output_dir / "results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    # Save histories
    for method, r in results.items():
        if "history" in r:
            with open(output_dir / f"history_{method}.json", "w") as f:
                json.dump(r["history"], f, indent=2)

    print(f"\nResults saved to {output_dir}")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    if dp_val is not None:
        for method in ["dp_shampoo", "dp_kfac", "dp_adadps", "dp_shampoo_synth", "dp_kfac_synth", "dp_adadps_synth"]:
            r = results.get(method, {})
            if "error" in r or metric_name not in r:
                continue
            val = r[metric_name]
            if val > dp_val + 0.005:
                print(f"  {method}: HELPS  ({val:.4f} vs {dp_val:.4f} dp)")
            elif val > dp_val - 0.005:
                print(f"  {method}: NEUTRAL ({val:.4f} vs {dp_val:.4f} dp)")
            else:
                print(f"  {method}: HURTS  ({val:.4f} vs {dp_val:.4f} dp)")


if __name__ == "__main__":
    main()

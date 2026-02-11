#!/usr/bin/env python3
"""
Hyperparameter tuning and full training for MICCAI experiments.

Supports:
- LR tuning for each method/dataset
- Full training with tuned or specified LR
- Multi-seed runs for statistical significance
- Epsilon sweep for privacy-utility tradeoff

Usage:
    # Tune LR then train
    python scripts/tune_and_train.py --dataset kvasir --phase both

    # Train with fixed LR
    python scripts/tune_and_train.py --dataset kvasir --phase train --lr 0.01

    # Multi-seed runs
    python scripts/tune_and_train.py --dataset kvasir --phase train --lr 0.01 --seeds 42 123 456

    # Epsilon sweep
    python scripts/tune_and_train.py --dataset kvasir --phase train --lr 0.01 --epsilons 1.0 2.0 4.0 8.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from opacus import GradSampleModule

from src.data import (
    MedMNISTDataset,
    KvasirSegDataset,
    OxfordPetSegmentation,
)
from src.models import create_model, create_classifier
from src.training import (
    DPTrainer,
    NonDPTrainer,
    DPClassificationTrainer,
    NonDPClassificationTrainer,
    AdaDPSPreconditioner,
    MomentumPreconditioner,
    ShampooPreconditioner,
    DiceBCELoss,
)

ALL_METHODS = ["baseline", "dp", "dp_synthetic", "dp_public", "dp_shampoo", "dp_shampoo_synth"]


def get_dataloaders(dataset_name, batch_size=32, image_size=128, seed=42):
    """Get train/val dataloaders for each dataset."""

    if dataset_name == "pathmnist":
        train_ds = MedMNISTDataset("pathmnist", split="train", download=True)
        val_ds = MedMNISTDataset("pathmnist", split="val", download=True)
        test_ds = MedMNISTDataset("pathmnist", split="test", download=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        task = "classification"
        num_classes = 9
        in_channels = 3

    elif dataset_name == "kvasir":
        train_ds = KvasirSegDataset(split="train", image_size=image_size, augment=True)
        val_ds = KvasirSegDataset(split="val", image_size=image_size, augment=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = None
        task = "segmentation"
        num_classes = 1
        in_channels = 3

    elif dataset_name == "pet":
        full_ds = OxfordPetSegmentation(split="trainval", image_size=image_size, augment=True)
        train_size = int(0.8 * len(full_ds))
        val_size = len(full_ds) - train_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(seed))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = None
        task = "segmentation"
        num_classes = 1
        in_channels = 3

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_loader, val_loader, test_loader, task, num_classes, in_channels


def create_model_for_task(task, in_channels, num_classes, model_size="medium"):
    """Create model based on task."""
    if task == "classification":
        if model_size == "small":
            return create_classifier("medmnist_cnn", in_channels, num_classes)
        else:
            return create_classifier("large_cnn", in_channels, num_classes)
    else:
        if model_size == "small":
            return create_model("unet", in_channels, 1, features=[32, 64, 128])
        else:
            return create_model("unet", in_channels, 1, features=[32, 64, 128, 256])


def run_experiment(dataset_name, method, lr, epochs, epsilon=8.0, damping=0.1,
                   device="cuda", batch_size=32, image_size=128, model_size="medium",
                   seed=42):
    """Run a single experiment."""

    torch.manual_seed(seed)

    # Get data
    train_loader, val_loader, test_loader, task, num_classes, in_channels = get_dataloaders(
        dataset_name, batch_size, image_size, seed
    )

    # Create model
    model = create_model_for_task(task, in_channels, num_classes, model_size)

    # Loss function
    if task == "classification":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = DiceBCELoss()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Setup preconditioner if needed
    preconditioner = None
    if method in ["dp_synthetic", "dp_momentum", "dp_public",
                   "dp_shampoo", "dp_shampoo_synth"]:
        if method == "dp_synthetic":
            preconditioner = AdaDPSPreconditioner(damping=damping, device=device)
        elif method == "dp_momentum":
            preconditioner = MomentumPreconditioner(damping=damping, beta2=0.9, device=device)
        elif method == "dp_public":
            preconditioner = AdaDPSPreconditioner(damping=damping, device=device)
        elif method in ["dp_shampoo", "dp_shampoo_synth"]:
            preconditioner = ShampooPreconditioner(damping=1e-4, device=device)

        wrapped = GradSampleModule(model).to(device)

        if method in ["dp_synthetic", "dp_momentum", "dp_shampoo_synth"]:
            preconditioner.estimate_from_synthetic(
                wrapped, loss_fn, num_samples=1000, batch_size=batch_size,
                image_size=image_size if task == "segmentation" else 28,
                in_channels=in_channels, num_classes=num_classes,
                task=task, mask_type="blob"
            )
        else:  # dp_public, dp_shampoo
            preconditioner.estimate_from_loader(
                wrapped, train_loader, loss_fn,
                num_steps=min(100, len(train_loader)), task=task
            )

        # Recreate model
        model = create_model_for_task(task, in_channels, num_classes, model_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Create trainer
    if task == "classification":
        if method == "baseline":
            trainer = NonDPClassificationTrainer(
                model, train_loader, val_loader, optimizer, loss_fn,
                epochs, device, scheduler=scheduler
            )
        else:
            trainer = DPClassificationTrainer(
                model, train_loader, val_loader, optimizer, loss_fn,
                epsilon, 1e-5, 1.0, epochs, device,
                preconditioner=preconditioner, scheduler=scheduler
            )
    else:
        if method == "baseline":
            trainer = NonDPTrainer(
                model, train_loader, val_loader, optimizer, loss_fn,
                epochs, device, scheduler=scheduler
            )
        else:
            trainer = DPTrainer(
                model, train_loader, val_loader, optimizer, loss_fn,
                epsilon, 1e-5, 1.0, epochs, device,
                preconditioner=preconditioner, scheduler=scheduler
            )

    # Train
    history = trainer.fit()

    # Get best metric
    if task == "classification":
        best_metric = max(history["val_acc"])
        final_metric = history["val_acc"][-1]
        metric_name = "val_acc"

        # Test set evaluation
        test_metric = None
        if test_loader is not None:
            trainer.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    if hasattr(trainer.model, '_module'):
                        outputs = trainer.model._module(images)
                    else:
                        outputs = trainer.model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            test_metric = 100.0 * correct / total
    else:
        best_metric = max(history["val_dice"])
        final_metric = history["val_dice"][-1]
        metric_name = "val_dice"
        test_metric = None

    return best_metric, final_metric, test_metric, history, metric_name


def tune_lr(dataset_name, method, epochs=5, device="cuda", epsilon=8.0):
    """Tune learning rate for a dataset/method combination."""

    lrs = [0.1, 0.05, 0.01, 0.005, 0.001]

    print(f"\n{'='*60}")
    print(f"Tuning LR for {dataset_name} / {method}")
    print(f"{'='*60}")

    results = []
    for lr in lrs:
        print(f"\n  LR={lr}...")
        try:
            best_metric, _, _, _, metric_name = run_experiment(
                dataset_name, method, lr, epochs, epsilon=epsilon, device=device
            )
            results.append((lr, best_metric))
            print(f"    {metric_name}: {best_metric:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append((lr, 0))

    # Find best
    best_lr, best_score = max(results, key=lambda x: x[1])
    print(f"\n  Best LR: {best_lr} ({metric_name}: {best_score:.4f})")

    return best_lr, results


def full_train(dataset_name, method, lr, epochs, epsilon=8.0, device="cuda",
               seed=42, output_dir="outputs"):
    """Full training run with best LR."""

    print(f"\n{'='*60}")
    print(f"Training: {dataset_name} / {method} / LR={lr} / eps={epsilon} / seed={seed}")
    print(f"{'='*60}")

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"miccai_{dataset_name}_{method}_s{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run
    best_metric, final_metric, test_metric, history, metric_name = run_experiment(
        dataset_name, method, lr, epochs, epsilon=epsilon, device=device, seed=seed
    )

    # Save
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    results = {
        "dataset": dataset_name,
        "method": method,
        "lr": lr,
        "epochs": epochs,
        "epsilon": epsilon if method != "baseline" else None,
        "seed": seed,
        f"best_{metric_name}": best_metric,
        f"final_{metric_name}": final_metric,
    }
    if test_metric is not None:
        results["test_acc"] = test_metric

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save config
    config = {
        "dataset": dataset_name,
        "method": method,
        "lr": lr,
        "epochs": epochs,
        "epsilon": epsilon,
        "seed": seed,
        "optimizer": "sgd",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "damping": 0.1,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Best {metric_name}: {best_metric:.4f}")
    if test_metric is not None:
        print(f"  Test accuracy: {test_metric:.2f}%")
    print(f"  Results saved to {run_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MICCAI experiment runner")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["pathmnist", "kvasir", "pet"])
    parser.add_argument("--phase", type=str, default="both",
                        choices=["tune", "train", "both"])
    parser.add_argument("--method", type=str, default="all",
                        choices=["all"] + ALL_METHODS)
    parser.add_argument("--lr", type=float, default=None,
                        help="Fixed LR (skip tuning)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tune_epochs", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=8.0,
                        help="Default epsilon (ignored if --epsilons is set)")
    parser.add_argument("--epsilons", type=float, nargs="+", default=None,
                        help="Epsilon sweep values (e.g., 1.0 2.0 4.0 8.0)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Seeds for multi-seed runs (e.g., 42 123 456 789 2024)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    methods = ALL_METHODS if args.method == "all" else [args.method]
    epsilons = args.epsilons or [args.epsilon]

    all_results = []

    for method in methods:
        # Determine LR
        if args.lr is not None:
            best_lr = args.lr
        elif args.phase in ["tune", "both"]:
            best_lr, _ = tune_lr(args.dataset, method, args.tune_epochs,
                                args.device, epsilon=epsilons[0])
        else:
            best_lr = 0.01  # fallback

        if args.phase in ["train", "both"]:
            for epsilon in epsilons:
                if method == "baseline" and epsilon != epsilons[0]:
                    continue  # baseline doesn't depend on epsilon

                for seed in args.seeds:
                    results = full_train(
                        args.dataset, method, best_lr, args.epochs,
                        epsilon=epsilon, device=args.device, seed=seed,
                        output_dir=args.output_dir
                    )
                    results["tuned_lr"] = best_lr
                    all_results.append(results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        method = r["method"]
        eps = r.get("epsilon", "-")
        seed = r.get("seed", 42)
        # Find the metric key
        metric_keys = [k for k in r if k.startswith("best_")]
        metric_val = r[metric_keys[0]] if metric_keys else "N/A"
        print(f"  {method:15s} eps={str(eps):5s} seed={seed}  best={metric_val:.4f}")

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(args.output_dir) / f"summary_miccai_{args.dataset}_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

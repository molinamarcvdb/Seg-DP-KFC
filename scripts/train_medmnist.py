#!/usr/bin/env python3
"""
Training script for MedMNIST classification experiments.

Supports:
- Baseline (no DP)
- DP-SGD without preconditioning
- DP-SGD with synthetic preconditioning (AdaDPS)

Usage:
    # Baseline (no DP)
    python scripts/train_medmnist.py --dataset pathmnist --method baseline

    # DP without preconditioning
    python scripts/train_medmnist.py --dataset pathmnist --method dp --epsilon 2.0

    # DP with synthetic preconditioning
    python scripts/train_medmnist.py --dataset pathmnist --method dp_synthetic --epsilon 2.0
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import GradSampleModule

from src.data import MedMNISTDataset, get_medmnist_info, SyntheticClassificationDataset
from src.models import create_classifier
from src.training import (
    DPClassificationTrainer,
    NonDPClassificationTrainer,
    AdaDPSPreconditioner,
    create_preconditioner,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train on MedMNIST")

    # Dataset
    parser.add_argument("--dataset", type=str, default="pathmnist",
                        help="MedMNIST dataset name")
    parser.add_argument("--data_root", type=str, default="./data/medmnist",
                        help="Data root directory")

    # Method
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "dp", "dp_synthetic", "dp_public"],
                        help="Training method")
    parser.add_argument("--public_dataset", type=str, default="dermamnist",
                        help="Public dataset for preconditioning (dp_public method)")

    # DP parameters
    parser.add_argument("--epsilon", type=float, default=2.0,
                        help="Privacy budget epsilon")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Privacy parameter delta")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm")

    # Preconditioning
    parser.add_argument("--precond_samples", type=int, default=500,
                        help="Number of synthetic samples for preconditioning")
    parser.add_argument("--noise_type", type=str, default="pink",
                        choices=["white", "pink", "brown", "perlin"],
                        help="Noise type for synthetic preconditioning")
    parser.add_argument("--damping", type=float, default=1e-4,
                        help="Damping for preconditioner")

    # Training
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")

    # Model
    parser.add_argument("--model", type=str, default="medmnist_cnn",
                        choices=["simple_cnn", "medmnist_cnn", "large_cnn"],
                        help="Model architecture")
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Base channels for model")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Get dataset info
    info = get_medmnist_info(args.dataset)
    num_classes = len(info["label"])
    print(f"Dataset: {args.dataset}")
    print(f"  Classes: {num_classes}")
    print(f"  Train size: {info['n_samples']['train']}")
    print(f"  Val size: {info['n_samples']['val']}")
    print(f"  Test size: {info['n_samples']['test']}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.dataset}_{args.method}"
    if args.method != "baseline":
        exp_name += f"_eps{args.epsilon}"
    output_dir = Path(args.output_dir) / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["num_classes"] = num_classes
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nOutput directory: {output_dir}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = MedMNISTDataset(
        args.dataset, split="train", download=True, root=args.data_root
    )
    val_dataset = MedMNISTDataset(
        args.dataset, split="val", download=True, root=args.data_root
    )
    test_dataset = MedMNISTDataset(
        args.dataset, split="test", download=True, root=args.data_root
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create model
    print(f"\nCreating model: {args.model}")
    model = create_classifier(
        name=args.model,
        in_channels=3,
        num_classes=num_classes,
        base_channels=args.base_channels,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Training
    print(f"\nMethod: {args.method}")

    if args.method == "baseline":
        # Non-DP training
        trainer = NonDPClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=args.epochs,
            device=args.device,
            scheduler=scheduler,
        )
        history = trainer.fit()

    elif args.method == "dp":
        # DP training without preconditioning
        print(f"  ε = {args.epsilon}, δ = {args.delta}")

        trainer = DPClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epsilon=args.epsilon,
            delta=args.delta,
            max_grad_norm=args.max_grad_norm,
            epochs=args.epochs,
            device=args.device,
            preconditioner=None,
            scheduler=scheduler,
        )
        history = trainer.fit()

    elif args.method == "dp_synthetic":
        # DP training with synthetic preconditioning
        print(f"  ε = {args.epsilon}, δ = {args.delta}")
        print(f"  Preconditioner: synthetic ({args.noise_type} noise)")

        # Create preconditioner
        preconditioner = AdaDPSPreconditioner(
            damping=args.damping,
            device=args.device,
        )

        # Wrap model for grad_sample computation
        wrapped_model = GradSampleModule(model)
        wrapped_model = wrapped_model.to(args.device)

        # Estimate preconditioner from synthetic data
        print(f"\nEstimating preconditioner from {args.precond_samples} synthetic samples...")
        preconditioner.estimate_from_synthetic(
            model=wrapped_model,
            loss_fn=loss_fn,
            num_samples=args.precond_samples,
            batch_size=args.batch_size,
            image_size=28,  # MedMNIST is 28x28
            in_channels=3,
            num_classes=num_classes,
            noise_type=args.noise_type,
            task="classification",
            show_progress=True,
        )

        # Save preconditioner
        preconditioner.save(str(output_dir / "preconditioner.pt"))
        print("  Preconditioner saved")

        # Create new model and optimizer (GradSampleModule was modified)
        model = create_classifier(
            name=args.model,
            in_channels=3,
            num_classes=num_classes,
            base_channels=args.base_channels,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        trainer = DPClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epsilon=args.epsilon,
            delta=args.delta,
            max_grad_norm=args.max_grad_norm,
            epochs=args.epochs,
            device=args.device,
            preconditioner=preconditioner,
            scheduler=scheduler,
        )
        history = trainer.fit()

    elif args.method == "dp_public":
        # DP training with public data preconditioning
        print(f"  ε = {args.epsilon}, δ = {args.delta}")
        print(f"  Preconditioner: public data ({args.public_dataset})")

        # Load public dataset
        public_dataset = MedMNISTDataset(
            args.public_dataset, split="train", download=True, root=args.data_root
        )
        public_loader = DataLoader(
            public_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        print(f"  Public dataset size: {len(public_dataset)}")

        # Create preconditioner
        preconditioner = AdaDPSPreconditioner(
            damping=args.damping,
            device=args.device,
        )

        # Wrap model for grad_sample computation
        wrapped_model = GradSampleModule(model)
        wrapped_model = wrapped_model.to(args.device)

        # Estimate preconditioner from public data
        num_steps = min(len(public_loader), args.precond_samples // args.batch_size)
        print(f"\nEstimating preconditioner from {num_steps} batches of public data...")
        preconditioner.estimate_from_loader(
            model=wrapped_model,
            data_loader=public_loader,
            loss_fn=loss_fn,
            num_steps=num_steps,
            show_progress=True,
            task="classification",
        )

        # Save preconditioner
        preconditioner.save(str(output_dir / "preconditioner.pt"))
        print("  Preconditioner saved")

        # Create new model and optimizer (GradSampleModule was modified)
        model = create_classifier(
            name=args.model,
            in_channels=3,
            num_classes=num_classes,
            base_channels=args.base_channels,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        trainer = DPClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epsilon=args.epsilon,
            delta=args.delta,
            max_grad_norm=args.max_grad_norm,
            epochs=args.epochs,
            device=args.device,
            preconditioner=preconditioner,
            scheduler=scheduler,
        )
        history = trainer.fit()

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    trainer.model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Handle wrapped model
            if hasattr(trainer.model, '_module'):
                outputs = trainer.model._module(images)
            else:
                outputs = trainer.model(images)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Save final results
    results = {
        "test_acc": test_acc,
        "best_val_acc": max(history["val_acc"]),
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    torch.save(
        trainer.model.state_dict() if not hasattr(trainer.model, '_module')
        else trainer.model._module.state_dict(),
        output_dir / "model.pt"
    )

    print(f"\nResults saved to {output_dir}")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Best Val Accuracy: {max(history['val_acc']):.2f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Training script for Oxford Pet segmentation experiments.

Supports:
- Baseline (no DP)
- DP-SGD without preconditioning
- DP-SGD with synthetic preconditioning
- DP-SGD with public data preconditioning

Usage:
    python scripts/train_pet_segmentation.py --method baseline
    python scripts/train_pet_segmentation.py --method dp --epsilon 8.0
    python scripts/train_pet_segmentation.py --method dp_synthetic --epsilon 8.0
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from opacus import GradSampleModule

from src.data import OxfordPetSegmentation
from src.models import create_model
from src.training import (
    DPTrainer,
    NonDPTrainer,
    AdaDPSPreconditioner,
    MomentumPreconditioner,
    ShampooPreconditioner,
    DiceBCELoss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train on Oxford Pet Segmentation")

    # Dataset
    parser.add_argument("--data_root", type=str, default="./data/oxford_pet",
                        help="Data root directory")
    parser.add_argument("--image_size", type=int, default=128,
                        help="Image size")

    # Method
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "dp", "dp_synthetic", "dp_momentum", "dp_public",
                                 "dp_shampoo", "dp_shampoo_synth"],
                        help="Training method")

    # DP parameters
    parser.add_argument("--epsilon", type=float, default=8.0,
                        help="Privacy budget epsilon")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Privacy parameter delta")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm")

    # Preconditioning
    parser.add_argument("--precond_samples", type=int, default=1000,
                        help="Number of samples for preconditioning")
    parser.add_argument("--shampoo_damping", type=float, default=1e-4,
                        help="Damping for Shampoo preconditioner")
    parser.add_argument("--noise_type", type=str, default="pink",
                        choices=["white", "pink", "brown", "perlin"],
                        help="Noise type for synthetic preconditioning")
    parser.add_argument("--damping", type=float, default=0.1,
                        help="Damping for preconditioner")
    parser.add_argument("--kfac_damping", type=float, default=1e-3,
                        help="Damping for K-FAC preconditioner")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="First moment decay for momentum preconditioner (0=disabled)")
    parser.add_argument("--beta2", type=float, default=0.9,
                        help="Second moment decay for momentum preconditioner")
    parser.add_argument("--adaptive_damping", action="store_true", default=True,
                        help="Use layer-wise adaptive damping")
    parser.add_argument("--mask_type", type=str, default="blob",
                        choices=["random", "blob", "perlin"],
                        help="Mask type for synthetic segmentation")

    # Training
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["adam", "sgd"],
                        help="Optimizer")

    # Model
    parser.add_argument("--model", type=str, default="unet",
                        help="Model architecture")
    parser.add_argument("--features", type=int, nargs="+", default=[32, 64, 128],
                        help="Feature channels for UNet")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    return parser.parse_args()


def compute_dice(pred, target, threshold=0.5):
    """Compute Dice coefficient."""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    return (2 * intersection) / (pred.sum() + target.sum() + 1e-8)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"pet_seg_{args.method}"
    if args.method != "baseline":
        exp_name += f"_eps{args.epsilon}"
    output_dir = Path(args.output_dir) / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Output directory: {output_dir}")

    # Load datasets
    print("\nLoading Oxford Pet dataset...")
    full_dataset = OxfordPetSegmentation(
        root=args.data_root,
        split="trainval",
        image_size=args.image_size,
        download=True,
        augment=True,
    )

    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Disable augmentation for validation
    val_dataset.dataset.augment = False

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    print(f"\nCreating model: {args.model}")
    model = create_model(
        name=args.model,
        in_channels=3,
        out_channels=1,
        features=args.features,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Loss function
    loss_fn = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)

    # Optimizer helper
    def make_optimizer(params):
        if args.optimizer == "sgd":
            return torch.optim.SGD(
                params, lr=args.lr, momentum=0.9, weight_decay=1e-4
            )
        else:
            return torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)

    optimizer = make_optimizer(model.parameters())
    print(f"Optimizer: {args.optimizer.upper()}")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Training
    print(f"\nMethod: {args.method}")

    if args.method == "baseline":
        trainer = NonDPTrainer(
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
        print(f"  epsilon = {args.epsilon}, delta = {args.delta}")

        trainer = DPTrainer(
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
        print(f"  epsilon = {args.epsilon}, delta = {args.delta}")
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
            image_size=args.image_size,
            in_channels=3,
            num_classes=1,  # Binary segmentation
            noise_type=args.noise_type,
            mask_type=args.mask_type,
            task="segmentation",
            show_progress=True,
        )

        # Save preconditioner
        preconditioner.save(str(output_dir / "preconditioner.pt"))
        print("  Preconditioner saved")

        # Create new model and optimizer
        model = create_model(
            name=args.model,
            in_channels=3,
            out_channels=1,
            features=args.features,
        )
        optimizer = make_optimizer(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        trainer = DPTrainer(
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

    elif args.method == "dp_momentum":
        print(f"  epsilon = {args.epsilon}, delta = {args.delta}")
        print(f"  Preconditioner: momentum (β1={args.beta1}, β2={args.beta2})")
        print(f"  Adaptive damping: {args.adaptive_damping}")

        # Create momentum preconditioner
        preconditioner = MomentumPreconditioner(
            damping=args.damping,
            beta1=args.beta1,
            beta2=args.beta2,
            adaptive_damping=args.adaptive_damping,
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
            image_size=args.image_size,
            in_channels=3,
            num_classes=1,
            noise_type=args.noise_type,
            mask_type=args.mask_type,
            task="segmentation",
            show_progress=True,
        )

        # Save preconditioner
        preconditioner.save(str(output_dir / "preconditioner.pt"))
        print("  Preconditioner saved")

        # Create new model and optimizer
        model = create_model(
            name=args.model,
            in_channels=3,
            out_channels=1,
            features=args.features,
        )
        optimizer = make_optimizer(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        trainer = DPTrainer(
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
        # DP with public data preconditioning (oracle - uses actual train data)
        print(f"  epsilon = {args.epsilon}, delta = {args.delta}")
        print(f"  Preconditioner: public data (oracle)")

        # Create preconditioner
        preconditioner = MomentumPreconditioner(
            damping=args.damping,
            beta1=args.beta1,
            beta2=args.beta2,
            adaptive_damping=args.adaptive_damping,
            device=args.device,
        )

        # Wrap model for grad_sample computation
        wrapped_model = GradSampleModule(model)
        wrapped_model = wrapped_model.to(args.device)

        # Estimate preconditioner from actual train data (oracle)
        num_steps = min(len(train_loader), args.precond_samples // args.batch_size)
        print(f"\nEstimating preconditioner from {num_steps} batches of train data...")
        preconditioner.estimate_from_loader(
            model=wrapped_model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            num_steps=num_steps,
            show_progress=True,
            task="segmentation",
        )

        # Save preconditioner
        preconditioner.save(str(output_dir / "preconditioner.pt"))
        print("  Preconditioner saved")

        # Create new model and optimizer
        model = create_model(
            name=args.model,
            in_channels=3,
            out_channels=1,
            features=args.features,
        )
        optimizer = make_optimizer(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        trainer = DPTrainer(
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

    elif args.method in ["dp_shampoo", "dp_shampoo_synth"]:
        is_public = args.method == "dp_shampoo"
        source = "train data (oracle)" if is_public else "synthetic data"
        print(f"  epsilon = {args.epsilon}, delta = {args.delta}")
        print(f"  Preconditioner: Shampoo from {source}")

        preconditioner = ShampooPreconditioner(
            damping=args.shampoo_damping, device=args.device
        )

        wrapped_model = GradSampleModule(model)
        wrapped_model = wrapped_model.to(args.device)

        if is_public:
            num_steps = min(100, len(train_loader))
            print(f"\nEstimating Shampoo from {num_steps} batches of train data...")
            preconditioner.estimate_from_loader(
                model=wrapped_model, data_loader=train_loader, loss_fn=loss_fn,
                num_steps=num_steps, show_progress=True, task="segmentation",
            )
        else:
            print(f"\nEstimating Shampoo from {args.precond_samples} synthetic samples...")
            preconditioner.estimate_from_synthetic(
                model=wrapped_model, loss_fn=loss_fn,
                num_samples=args.precond_samples, batch_size=args.batch_size,
                image_size=args.image_size, in_channels=3, num_classes=1,
                noise_type=args.noise_type, mask_type=args.mask_type,
                task="segmentation", show_progress=True,
            )

        preconditioner.save(str(output_dir / "preconditioner.pt"))
        print("  Preconditioner saved")

        model = create_model(
            name=args.model, in_channels=3, out_channels=1, features=args.features,
        )
        optimizer = make_optimizer(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        trainer = DPTrainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, loss_fn=loss_fn,
            epsilon=args.epsilon, delta=args.delta,
            max_grad_norm=args.max_grad_norm, epochs=args.epochs,
            device=args.device, preconditioner=preconditioner,
            scheduler=scheduler,
        )
        history = trainer.fit()

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation
    print("\nFinal Results:")
    print(f"  Best Val Dice: {max(history['val_dice']):.4f}")
    print(f"  Final Val Dice: {history['val_dice'][-1]:.4f}")

    # Save results
    results = {
        "dataset": "pet",
        "method": args.method,
        "epsilon": args.epsilon if args.method != "baseline" else None,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "best_val_dice": max(history["val_dice"]),
        "final_val_dice": history["val_dice"][-1],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

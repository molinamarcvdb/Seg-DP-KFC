#!/usr/bin/env python3
"""
Training script for Retinal Vessel Segmentation experiments.

Supports:
- Baseline (no DP)
- DP-SGD without preconditioning
- DP-SGD with synthetic preconditioning (AdaDPS)
- DP-SGD with public data preconditioning (oracle)
- DP-SGD with Shampoo preconditioning (public oracle)
- DP-SGD with Shampoo preconditioning (synthetic)

Usage:
    python scripts/train_fundus.py --method baseline --epochs 20
    python scripts/train_fundus.py --method dp --epsilon 8.0 --epochs 20
    python scripts/train_fundus.py --method dp_shampoo_synth --epsilon 8.0 --epochs 20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from opacus import GradSampleModule

from src.data.retinal_vessel_dataset import RetinalVesselDataset
from src.models import create_model
from src.training import (
    DPTrainer,
    NonDPTrainer,
    AdaDPSPreconditioner,
    ShampooPreconditioner,
    DiceBCELoss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train on Retinal Vessel Segmentation")

    # Method
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "dp", "dp_synthetic",
                                 "dp_public", "dp_shampoo", "dp_shampoo_synth"])

    # DP parameters
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])

    # Preconditioning
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--shampoo_damping", type=float, default=1e-4)
    parser.add_argument("--precond_samples", type=int, default=1000)
    parser.add_argument("--noise_type", type=str, default="pink",
                        choices=["white", "pink", "brown", "perlin"])
    parser.add_argument("--mask_type", type=str, default="blob",
                        choices=["random", "blob", "perlin"])
    parser.add_argument("--mask_strategy", type=str, default="frangi",
                        choices=["gaussian_blobs", "random_shapes", "pink_threshold",
                                 "voronoi", "frangi"])

    # Model
    parser.add_argument("--features", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--image_size", type=int, default=128)

    # Data
    parser.add_argument("--data_root", type=str, default="./data/archive-2")
    parser.add_argument("--subset_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (0-1)")
    parser.add_argument("--val_every", type=int, default=0,
                        help="Validate every N epochs (0=every epoch)")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"fundus_{args.method}"
    if args.method != "baseline":
        exp_name += f"_eps{args.epsilon}"
    if args.subset_fraction < 1.0:
        exp_name += f"_f{args.subset_fraction}"
    output_dir = Path(args.output_dir) / f"{exp_name}_s{args.seed}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Output directory: {output_dir}")

    # Data
    print("\nLoading Retinal Vessel dataset...")
    train_dataset = RetinalVesselDataset(
        root=args.data_root, split="train", image_size=args.image_size,
        augment=True, seed=args.seed,
    )
    val_dataset = RetinalVesselDataset(
        root=args.data_root, split="val", image_size=args.image_size,
        augment=False, seed=args.seed,
    )
    # Subset training data if requested (val stays full for fair comparison)
    if args.subset_fraction < 1.0:
        n_train = max(1, int(len(train_dataset.pairs) * args.subset_fraction))
        train_dataset.pairs = train_dataset.pairs[:n_train]
        print(f"  Using {args.subset_fraction:.0%} of training data: {n_train} samples")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = create_model("unet", in_channels=3, out_channels=1, features=args.features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Loss & Optimizer
    loss_fn = DiceBCELoss()

    def make_optimizer(params):
        if args.optimizer == "sgd":
            return torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
        else:
            return torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)

    optimizer = make_optimizer(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print(f"\nMethod: {args.method}")
    print(f"Optimizer: {args.optimizer.upper()}")

    if args.method == "baseline":
        trainer = NonDPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                               args.epochs, args.device, scheduler=scheduler)

    elif args.method == "dp":
        print(f"  epsilon={args.epsilon}")
        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           args.epsilon, args.delta, args.max_grad_norm,
                           args.epochs, args.device, preconditioner=None,
                           scheduler=scheduler)

    elif args.method == "dp_synthetic":
        print(f"  epsilon={args.epsilon}, preconditioning from synthetic data")

        precond = AdaDPSPreconditioner(damping=args.damping, device=args.device)
        wrapped = GradSampleModule(model).to(args.device)
        precond.estimate_from_synthetic(
            wrapped, loss_fn, num_samples=args.precond_samples,
            batch_size=args.batch_size, image_size=args.image_size,
            in_channels=3, num_classes=1, noise_type=args.noise_type,
            task="segmentation", mask_type=args.mask_type,
            mask_strategy=args.mask_strategy, spatial_dims=2)
        precond.save(str(output_dir / "preconditioner.pt"))

        model = create_model("unet", in_channels=3, out_channels=1, features=args.features)
        optimizer = make_optimizer(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           args.epsilon, args.delta, args.max_grad_norm,
                           args.epochs, args.device, preconditioner=precond,
                           scheduler=scheduler)

    elif args.method == "dp_public":
        print(f"  epsilon={args.epsilon}, preconditioning from train data (oracle)")

        precond = AdaDPSPreconditioner(damping=args.damping, device=args.device)
        wrapped = GradSampleModule(model).to(args.device)
        num_steps = min(50, len(train_loader))
        precond.estimate_from_loader(wrapped, train_loader, loss_fn,
                                    num_steps=num_steps, task="segmentation")
        precond.save(str(output_dir / "preconditioner.pt"))

        model = create_model("unet", in_channels=3, out_channels=1, features=args.features)
        optimizer = make_optimizer(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           args.epsilon, args.delta, args.max_grad_norm,
                           args.epochs, args.device, preconditioner=precond,
                           scheduler=scheduler)

    elif args.method in ["dp_shampoo", "dp_shampoo_synth"]:
        is_public = args.method == "dp_shampoo"
        source = "train data (oracle)" if is_public else "synthetic data"
        print(f"  epsilon={args.epsilon}, Shampoo preconditioning from {source}")

        precond = ShampooPreconditioner(damping=args.shampoo_damping, device=args.device)
        wrapped = GradSampleModule(model).to(args.device)

        if is_public:
            num_steps = min(100, len(train_loader))
            precond.estimate_from_loader(wrapped, train_loader, loss_fn,
                                        num_steps=num_steps, task="segmentation")
        else:
            precond.estimate_from_synthetic(
                wrapped, loss_fn, num_samples=args.precond_samples,
                batch_size=args.batch_size, image_size=args.image_size,
                in_channels=3, num_classes=1, noise_type=args.noise_type,
                task="segmentation", mask_type=args.mask_type,
                mask_strategy=args.mask_strategy, spatial_dims=2)
        precond.save(str(output_dir / "preconditioner.pt"))

        model = create_model("unet", in_channels=3, out_channels=1, features=args.features)
        optimizer = make_optimizer(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           args.epsilon, args.delta, args.max_grad_norm,
                           args.epochs, args.device, preconditioner=precond,
                           scheduler=scheduler)

    # Training loop with optional sparse validation
    if args.val_every > 0:
        history = {"train_loss": [], "val_loss": [], "val_dice": []}
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            train_metrics = trainer.train_epoch()
            history["train_loss"].append(train_metrics["train_loss"])
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")

            if (epoch + 1) % args.val_every == 0 or (epoch + 1) == args.epochs:
                val_metrics = trainer.validate()
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_dice"].append(val_metrics["val_dice"])
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Dice: {val_metrics['val_dice']:.4f}")

            if trainer.scheduler is not None:
                trainer.scheduler.step()
    else:
        history = trainer.fit()

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Results
    results = {
        "dataset": "fundus",
        "target": "vessel",
        "method": args.method,
        "epsilon": args.epsilon if args.method != "baseline" else None,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "subset_fraction": args.subset_fraction,
        "n_train": len(train_dataset),
        "best_val_dice": max(history["val_dice"]),
        "final_val_dice": history["val_dice"][-1],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
    }

    print(f"\nResults:")
    print(f"  Best Val Dice: {results['best_val_dice']:.4f}")
    print(f"  Final Val Dice: {results['final_val_dice']:.4f}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

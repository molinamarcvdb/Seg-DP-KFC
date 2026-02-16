#!/usr/bin/env python3
"""
Training script for Hepatic Vessel Segmentation experiments (MSD Task08).

Supports:
- Baseline (no DP)
- DP-SGD without preconditioning
- DP-SGD with synthetic preconditioning (AdaDPS)
- DP-SGD with public data preconditioning (oracle)
- DP-SGD with Shampoo preconditioning (public oracle)
- DP-SGD with Shampoo preconditioning (synthetic)

Usage:
    python scripts/train_hepatic.py --method baseline --epochs 10
    python scripts/train_hepatic.py --method dp --epsilon 8.0 --epochs 10
    python scripts/train_hepatic.py --method dp_shampoo_synth --epsilon 8.0 --epochs 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import argparse
import json
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from opacus import GradSampleModule

from src.data.hepatic_dataset import HepaticVesselDataset, HEPATIC_DEFAULT_ROOT
from src.models.unet3d import create_model_3d
from src.training import (
    DPTrainer,
    NonDPTrainer,
    AdaDPSPreconditioner,
    ShampooPreconditioner,
    DiceBCELoss,
)


def sliding_window_inference(model, volume, patch_size, device, out_channels=1, batch_size=4):
    """
    Sliding window inference on a full 3D volume.

    Args:
        model: Trained model
        volume: (C_in, H, W, D) tensor
        patch_size: Patch size for inference
        device: Device
        out_channels: Number of output channels
        batch_size: Number of patches per forward pass

    Returns:
        prediction: (C_out, H, W, D) probability map
    """
    model.eval()
    C_in = volume.shape[0]
    _, H, W, D = volume.shape
    stride = patch_size // 2

    pred_sum = torch.zeros(out_channels, H, W, D, device='cpu')
    count = torch.zeros(out_channels, H, W, D, device='cpu')

    coords = []
    for h0 in range(0, max(1, H - patch_size + 1), stride):
        for w0 in range(0, max(1, W - patch_size + 1), stride):
            for d0 in range(0, max(1, D - patch_size + 1), stride):
                h0 = min(h0, max(0, H - patch_size))
                w0 = min(w0, max(0, W - patch_size))
                d0 = min(d0, max(0, D - patch_size))
                coords.append((h0, w0, d0))

    coords = list(set(coords))

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i+batch_size]
            patches = []
            for (h0, w0, d0) in batch_coords:
                patch = volume[:, h0:h0+patch_size, w0:w0+patch_size, d0:d0+patch_size]
                _, ph, pw, pd = patch.shape
                if ph < patch_size or pw < patch_size or pd < patch_size:
                    padded = torch.zeros(C_in, patch_size, patch_size, patch_size)
                    padded[:, :ph, :pw, :pd] = patch
                    patch = padded
                patches.append(patch)

            batch_tensor = torch.stack(patches).to(device)
            out = torch.sigmoid(model(batch_tensor)).cpu()

            for j, (h0, w0, d0) in enumerate(batch_coords):
                h_end = min(h0 + patch_size, H)
                w_end = min(w0 + patch_size, W)
                d_end = min(d0 + patch_size, D)
                pred_sum[:, h0:h_end, w0:w_end, d0:d_end] += out[j, :, :h_end-h0, :w_end-w0, :d_end-d0]
                count[:, h0:h_end, w0:w_end, d0:d_end] += 1

    count = torch.clamp(count, min=1)
    return pred_sum / count


def evaluate_full_volumes(model, dataset, patch_size, device, out_channels=1,
                          channel_names=None, n_volumes=None):
    """Evaluate model on full validation volumes using sliding window."""
    n_cases = len(dataset.cases)
    if n_volumes is not None:
        n_cases = min(n_cases, n_volumes)

    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(out_channels)]

    # Collect predictions and masks for threshold search
    all_preds = []
    all_masks = []
    
    print(f"  Running inference on {n_cases} volumes...")
    for i in range(n_cases):
        volume, mask = dataset.get_full_volume(i)
        pred = sliding_window_inference(model, volume, patch_size, device,
                                        out_channels=out_channels)
        all_preds.append(pred)
        all_masks.append(mask)
    
    # Search for optimal threshold
    best_thresh = 0.5
    best_mean_dice = 0.0
    
    print("  Searching for optimal threshold...")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        thresh_dices = {name: [] for name in channel_names}
        for pred, mask in zip(all_preds, all_masks):
            pred_binary = (pred > thresh).float()
            for c, name in enumerate(channel_names):
                p = pred_binary[c]
                m = mask[c]
                intersection = (p * m).sum()
                union = p.sum() + m.sum()
                dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
                thresh_dices[name].append(dice.item())
        
        mean_dice = np.mean([np.mean(thresh_dices[n]) for n in channel_names])
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            best_thresh = thresh
    
    print(f"  Best threshold: {best_thresh}")
    
    # Compute final results with best threshold
    all_dices = {name: [] for name in channel_names}
    for i, (pred, mask) in enumerate(zip(all_preds, all_masks)):
        pred_binary = (pred > best_thresh).float()
        vol_dices = []
        for c, name in enumerate(channel_names):
            p = pred_binary[c]
            m = mask[c]
            intersection = (p * m).sum()
            union = p.sum() + m.sum()
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
            all_dices[name].append(dice.item())
            vol_dices.append(dice.item())

        mean_vol = np.mean(vol_dices)
        detail = ", ".join(f"{n}={d:.4f}" for n, d in zip(channel_names, vol_dices))
        print(f"  Volume {i+1}/{n_cases}: Dice={mean_vol:.4f} ({detail})")

    results = {"best_threshold": best_thresh}
    for name in channel_names:
        results[f"dice_{name}"] = np.mean(all_dices[name])
    results["mean_dice"] = np.mean([results[f"dice_{n}"] for n in channel_names])

    detail = ", ".join(f"{n}={results[f'dice_{n}']:.4f}" for n in channel_names)
    print(f"  Mean Dice: {results['mean_dice']:.4f} ({detail})")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train on Hepatic Vessel (MSD Task08)")

    # Method
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "dp", "dp_synthetic",
                                 "dp_public", "dp_shampoo", "dp_shampoo_synth"])

    # DP parameters
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])

    # Preconditioning
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--shampoo_damping", type=float, default=1e-4)
    parser.add_argument("--precond_samples", type=int, default=1000)
    parser.add_argument("--noise_type", type=str, default="pink",
                        choices=["white", "pink", "brown", "perlin"])
    parser.add_argument("--mask_strategy", type=str, default="frangi",
                        choices=["gaussian_blobs", "random_shapes", "pink_threshold",
                                 "voronoi", "frangi", "vessel_tumor_combo"])

    # Model
    parser.add_argument("--features", type=int, nargs="+", default=[16, 32, 64, 128])

    # Data
    parser.add_argument("--data_root", type=str, default=HEPATIC_DEFAULT_ROOT)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--estimation_patch_size", type=int, default=128)
    parser.add_argument("--patches_per_volume", type=int, default=16)
    parser.add_argument("--target", type=str, default="all",
                        choices=["multilabel", "vessel", "tumour", "all"])

    # Performance
    parser.add_argument("--num_workers", type=int, default=4)

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_volumes", type=int, default=None,
                        help="Number of validation volumes for sliding window eval (None=all)")

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"hepatic_{args.method}"
    if args.method != "baseline":
        exp_name += f"_eps{args.epsilon}"
    output_dir = Path(args.output_dir) / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Output directory: {output_dir}")

    # Data
    print("\nLoading Hepatic Vessel (Task08)...")
    train_dataset = HepaticVesselDataset(
        root=args.data_root, split="train", patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume, target=args.target, augment=True,
        seed=args.seed,
    )
    val_dataset = HepaticVesselDataset(
        root=args.data_root, split="val", patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume, target=args.target, augment=False,
        seed=args.seed,
    )

    nw = args.num_workers
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    print(f"  Train: {len(train_dataset)} patches ({len(train_dataset.cases)} volumes)")
    print(f"  Val: {len(val_dataset)} patches ({len(val_dataset.cases)} volumes)")

    # Model (1 input channel for CT)
    out_channels = 2 if args.target == "multilabel" else 1
    channel_names = ["vessel", "tumour"] if args.target == "multilabel" else ["foreground"]
    model = create_model_3d("unet3d", in_channels=1, out_channels=out_channels, features=args.features)
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
        print(f"  epsilon={args.epsilon}, AdaDPS preconditioning from synthetic data")
        print(f"  noise_type={args.noise_type}, mask_strategy={args.mask_strategy}")

        precond = AdaDPSPreconditioner(damping=args.damping, device=args.device)
        wrapped = GradSampleModule(model).to(args.device)

        precond.estimate_from_synthetic(
            wrapped, loss_fn, num_samples=args.precond_samples,
            batch_size=args.batch_size, image_size=args.estimation_patch_size,
            in_channels=1, num_classes=out_channels, noise_type=args.noise_type,
            mask_strategy=args.mask_strategy, task="segmentation", spatial_dims=3,
        )
        precond.save(str(output_dir / "preconditioner.pt"))

        model = create_model_3d("unet3d", in_channels=1, out_channels=out_channels, features=args.features)
        optimizer = make_optimizer(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           args.epsilon, args.delta, args.max_grad_norm,
                           args.epochs, args.device, preconditioner=precond,
                           scheduler=scheduler)

    elif args.method == "dp_public":
        print(f"  epsilon={args.epsilon}, AdaDPS preconditioning from train data (oracle)")

        precond = AdaDPSPreconditioner(damping=args.damping, device=args.device)
        wrapped = GradSampleModule(model).to(args.device)
        num_steps = min(100, len(train_loader))
        precond.estimate_from_loader(wrapped, train_loader, loss_fn,
                                    num_steps=num_steps, task="segmentation")
        precond.save(str(output_dir / "preconditioner.pt"))

        model = create_model_3d("unet3d", in_channels=1, out_channels=out_channels, features=args.features)
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
        if not is_public:
            print(f"  noise_type={args.noise_type}, mask_strategy={args.mask_strategy}")

        precond = ShampooPreconditioner(damping=args.shampoo_damping, device=args.device)
        wrapped = GradSampleModule(model).to(args.device)

        if is_public:
            num_steps = min(100, len(train_loader))
            precond.estimate_from_loader(wrapped, train_loader, loss_fn,
                                        num_steps=num_steps, task="segmentation")
        else:
            precond.estimate_from_synthetic(
                wrapped, loss_fn, num_samples=args.precond_samples,
                batch_size=args.batch_size, image_size=args.estimation_patch_size,
                in_channels=1, num_classes=out_channels, noise_type=args.noise_type,
                mask_strategy=args.mask_strategy, task="segmentation", spatial_dims=3,
            )
        precond.save(str(output_dir / "preconditioner.pt"))

        model = create_model_3d("unet3d", in_channels=1, out_channels=out_channels, features=args.features)
        optimizer = make_optimizer(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           args.epsilon, args.delta, args.max_grad_norm,
                           args.epochs, args.device, preconditioner=precond,
                           scheduler=scheduler)

    # Train with periodic checkpointing
    checkpoint_every = 10  # Save checkpoint every N epochs
    best_val_dice = 0.0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.validate()
        if scheduler is not None:
            scheduler.step()

        train_loss = train_metrics["train_loss"]
        val_loss = val_metrics["val_loss"]
        val_dice = val_metrics["val_dice"]

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice

        # Periodic checkpoint
        if epoch % checkpoint_every == 0 or epoch == args.epochs:
            interim_results = {
                "dataset": "hepatic",
                "target": args.target,
                "method": args.method,
                "epsilon": args.epsilon if args.method != "baseline" else None,
                "lr": args.lr,
                "epochs_completed": epoch,
                "epochs_total": args.epochs,
                "seed": args.seed,
                "patch_size": args.patch_size,
                "best_val_dice": best_val_dice,
                "current_val_dice": val_dice,
                "current_train_loss": train_loss,
                "current_val_loss": val_loss,
                "status": "running" if epoch < args.epochs else "completed",
            }
            with open(output_dir / "results.json", "w") as f:
                json.dump(interim_results, f, indent=2)
            with open(output_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            print(f"  Checkpoint saved (epoch {epoch})", flush=True)

    # Patch-based results
    results = {
        "dataset": "hepatic",
        "target": args.target,
        "method": args.method,
        "epsilon": args.epsilon if args.method != "baseline" else None,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "patch_size": args.patch_size,
        "best_val_dice": best_val_dice,
        "final_val_dice": history["val_dice"][-1],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
    }

    # Sliding window inference on full validation volumes
    print("\nSliding window inference on validation volumes...")
    model_for_eval = trainer.model
    if hasattr(model_for_eval, '_module'):
        model_for_eval = model_for_eval._module
    model_for_eval.eval()
    vol_results = evaluate_full_volumes(
        model_for_eval, val_dataset, args.patch_size, args.device,
        out_channels=out_channels, channel_names=channel_names,
        n_volumes=args.eval_volumes,
    )
    results["full_volume_dice"] = vol_results["mean_dice"]
    results["full_volume_dice_per_channel"] = {
        k: v for k, v in vol_results.items() if k != "mean_dice"
    }
    results["status"] = "completed"

    print(f"\nResults:")
    print(f"  Best Val Dice (patches): {results['best_val_dice']:.4f}")
    print(f"  Full Volume Dice: {results['full_volume_dice']:.4f}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

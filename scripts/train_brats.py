#!/usr/bin/env python3
"""
Training script for BraTS 2025 brain tumor segmentation experiments.

Supports:
- Baseline (no DP)
- DP-SGD without preconditioning
- DP-SGD with synthetic preconditioning (AdaDPS)
- DP-SGD with public data preconditioning (oracle)
- DP-SGD with Shampoo preconditioning (public oracle)
- DP-SGD with Shampoo preconditioning (synthetic)

Usage:
    python scripts/train_brats.py --method baseline --epochs 5
    python scripts/train_brats.py --method dp --epsilon 8.0 --epochs 5
    python scripts/train_brats.py --method dp_shampoo_synth --epsilon 8.0 --epochs 5
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
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import GradSampleModule

from src.data.brats_dataset import BraTSPatchDataset, BRATS_DEFAULT_ROOT
from src.models.unet3d import create_model_3d
from src.training import (
    DPTrainer,
    NonDPTrainer,
    AdaDPSPreconditioner,
    ShampooPreconditioner,
    DiceBCELoss,
)


# Global helper for multiprocessing (must be at module level to be picklable)
def _load_volume_for_cache(args_tuple):
    """Load a single volume for parallel caching (module-level function for pickling)."""
    case_path, use_npz, target = args_tuple
    from src.data.brats_dataset import _load_volume_npz, _load_volume, _binarize_target
    
    if use_npz:
        image, seg = _load_volume_npz(case_path)
    else:
        image, seg = _load_volume(case_path)
    
    mask = _binarize_target(seg, target)
    return str(case_path), (image, mask, None)


def sliding_window_inference(model, volume, patch_size, device, out_channels=1, batch_size=4):
    """
    Sliding window inference on a full 3D volume.

    Args:
        model: Trained model
        volume: (C_in, D, H, W) tensor
        patch_size: Patch size for inference
        device: Device
        out_channels: Number of output channels
        batch_size: Number of patches per forward pass

    Returns:
        prediction: (C_out, D, H, W) probability map
    """
    model.eval()
    C_in = volume.shape[0]
    _, D, H, W = volume.shape
    stride = patch_size // 2  # 50% overlap

    pred_sum = torch.zeros(out_channels, D, H, W, device='cpu')
    count = torch.zeros(out_channels, D, H, W, device='cpu')

    # Collect all patch coordinates
    coords = []
    for d0 in range(0, max(1, D - patch_size + 1), stride):
        for h0 in range(0, max(1, H - patch_size + 1), stride):
            for w0 in range(0, max(1, W - patch_size + 1), stride):
                d0 = min(d0, max(0, D - patch_size))
                h0 = min(h0, max(0, H - patch_size))
                w0 = min(w0, max(0, W - patch_size))
                coords.append((d0, h0, w0))

    # Remove duplicates
    coords = list(set(coords))

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i+batch_size]
            patches = []
            for (d0, h0, w0) in batch_coords:
                patch = volume[:, d0:d0+patch_size, h0:h0+patch_size, w0:w0+patch_size]
                # Pad if necessary
                _, pd, ph, pw = patch.shape
                if pd < patch_size or ph < patch_size or pw < patch_size:
                    padded = torch.zeros(C_in, patch_size, patch_size, patch_size)
                    padded[:, :pd, :ph, :pw] = patch
                    patch = padded
                patches.append(patch)

            batch_tensor = torch.stack(patches).to(device)
            out = torch.sigmoid(model(batch_tensor)).cpu()

            for j, (d0, h0, w0) in enumerate(batch_coords):
                d_end = min(d0 + patch_size, D)
                h_end = min(h0 + patch_size, H)
                w_end = min(w0 + patch_size, W)
                pred_sum[:, d0:d_end, h0:h_end, w0:w_end] += out[j, :, :d_end-d0, :h_end-h0, :w_end-w0]
                count[:, d0:d_end, h0:h_end, w0:w_end] += 1

    count = torch.clamp(count, min=1)
    return pred_sum / count


def evaluate_full_volumes(model, dataset, patch_size, device, out_channels=1,
                          channel_names=None, n_volumes=None):
    """
    Evaluate model on full validation volumes using sliding window.

    Returns:
        dict with 'mean_dice' (overall) and per-channel Dice if multi-label
    """
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
    
    # Search for optimal threshold (like training validation does)
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
    parser = argparse.ArgumentParser(description="Train on BraTS 2025")

    # Method
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "dp", "dp_synthetic",
                                 "dp_public", "dp_shampoo", "dp_shampoo_synth"])

    # DP parameters
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])

    # Preconditioning
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--shampoo_damping", type=float, default=1e-4)
    parser.add_argument("--precond_samples", type=int, default=1000)
    parser.add_argument("--noise_type", type=str, default="pink",
                        choices=["white", "pink", "brown", "perlin"])
    parser.add_argument("--mask_strategy", type=str, default="nested_blobs",
                        choices=["gaussian_blobs", "random_shapes", "pink_threshold",
                                 "voronoi", "frangi", "nested_blobs"])

    # Model
    parser.add_argument("--features", type=int, nargs="+", default=[16, 32, 64, 128])

    # Data
    parser.add_argument("--data_root", type=str, default=BRATS_DEFAULT_ROOT)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--estimation_patch_size", type=int, default=64)
    parser.add_argument("--patches_per_volume", type=int, default=4)
    parser.add_argument("--target", type=str, default="multilabel",
                        choices=["multilabel", "wt", "tc", "et"])

    # Performance
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_volumes", type=int, default=8,
                        help="Number of volumes to cache in RAM per dataset")
    parser.add_argument("--cache_all", action="store_true",
                        help="Preload ALL volumes into RAM for maximum speed")
    parser.add_argument("--subset_fraction", type=float, default=1.0,
                        help="Use only this fraction of the dataset (0-1)")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_volumes", type=int, default=None,
                        help="Number of validation volumes for sliding window eval (None=all)")
    parser.add_argument("--skip_volume_eval", action="store_true",
                        help="Skip sliding window full volume evaluation")

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"brats_{args.method}"
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
    print("\nLoading BraTS 2025...")
    
    # Determine cache size
    if args.cache_all:
        print("⚡ CACHE_ALL enabled: Preloading ALL volumes into RAM for maximum speed")
        cache_size = 10000  # Effectively unlimited
    else:
        cache_size = args.cache_volumes
    
    train_dataset = BraTSPatchDataset(
        root=args.data_root, split="train", patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume, target=args.target, augment=True,
        seed=args.seed, cache_volumes=cache_size,
    )
    val_dataset = BraTSPatchDataset(
        root=args.data_root, split="val", patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume, target=args.target, augment=False,
        seed=args.seed, cache_volumes=cache_size,
    )

    # Subset if requested
    if args.subset_fraction < 1.0:
        n_train = max(1, int(len(train_dataset.cases) * args.subset_fraction))
        n_val = max(1, int(len(val_dataset.cases) * args.subset_fraction))
        train_dataset.cases = train_dataset.cases[:n_train]
        val_dataset.cases = val_dataset.cases[:n_val]
    
    # Prefetch all volumes into cache if cache_all enabled
    if args.cache_all:
        from tqdm import tqdm
        import time
        
        # Simple sequential loading with tqdm (most reliable)
        # npz files are already preprocessed, so this is reasonably fast
        print(f"\n⚡ Prefetching {len(train_dataset.cases)} training volumes into RAM...")
        start_time = time.time()
        
        for case in tqdm(train_dataset.cases, desc="Loading train volumes", unit="vol", ncols=100):
            train_dataset._get_volume(case)
        
        train_time = time.time() - start_time
        train_rate = len(train_dataset.cases) / train_time if train_time > 0 else 0
        print(f"✓ Training cache: {len(train_dataset._cache)} volumes in RAM ({train_time:.1f}s, {train_rate:.1f} vol/s)")
        
        print(f"\n⚡ Prefetching {len(val_dataset.cases)} validation volumes into RAM...")
        start_time = time.time()
        
        for case in tqdm(val_dataset.cases, desc="Loading val volumes", unit="vol", ncols=100):
            val_dataset._get_volume(case)
        
        val_time = time.time() - start_time
        val_rate = len(val_dataset.cases) / val_time if val_time > 0 else 0
        print(f"✓ Validation cache: {len(val_dataset._cache)} volumes in RAM ({val_time:.1f}s, {val_rate:.1f} vol/s)")
        
        total_vols = len(train_dataset._cache) + len(val_dataset._cache)
        total_time = train_time + val_time
        print(f"✓ Total: {total_vols} volumes cached in {total_time:.1f}s ({total_vols/total_time:.1f} vol/s)\n")
        
        # Set num_workers to 0 since all data is in RAM
        nw = 0
        print("⚡ DataLoader num_workers=0 (all data in RAM, zero I/O overhead)")
    else:
        nw = args.num_workers
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    print(f"  Train: {len(train_dataset)} patches ({len(train_dataset.cases)} volumes)")
    print(f"  Val: {len(val_dataset)} patches ({len(val_dataset.cases)} volumes)")

    # Model
    out_channels = 3 if args.target == "multilabel" else 1
    channel_names = ["WT", "TC", "ET"] if args.target == "multilabel" else [args.target.upper()]
    model = create_model_3d("unet3d", in_channels=4, out_channels=out_channels, features=args.features)
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
            in_channels=4, num_classes=out_channels, noise_type=args.noise_type,
            mask_strategy=args.mask_strategy, task="segmentation", spatial_dims=3,
        )
        precond.save(str(output_dir / "preconditioner.pt"))

        model = create_model_3d("unet3d", in_channels=4, out_channels=out_channels, features=args.features)
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

        model = create_model_3d("unet3d", in_channels=4, out_channels=out_channels, features=args.features)
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
                in_channels=4, num_classes=out_channels, noise_type=args.noise_type,
                mask_strategy=args.mask_strategy, task="segmentation", spatial_dims=3,
            )
        precond.save(str(output_dir / "preconditioner.pt"))

        model = create_model_3d("unet3d", in_channels=4, out_channels=out_channels, features=args.features)
        optimizer = make_optimizer(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        trainer = DPTrainer(model, train_loader, val_loader, optimizer, loss_fn,
                           args.epsilon, args.delta, args.max_grad_norm,
                           args.epochs, args.device, preconditioner=precond,
                           scheduler=scheduler)

    # Save best model via callbacks
    _best_dice = [0.0]
    _last_metrics = [{}]

    def _log(epoch, metrics):
        _last_metrics[0] = metrics

    def _save_best(epoch, model):
        val_dice = _last_metrics[0].get("val_dice", 0.0)
        if val_dice > _best_dice[0]:
            _best_dice[0] = val_dice
            m = model._module if hasattr(model, '_module') else model
            torch.save(m.state_dict(), output_dir / "model_best.pt")
            print(f"  Saved best model (dice={val_dice:.4f})")

    history = trainer.fit(log_fn=_log, save_fn=_save_best)

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Patch-based results
    results = {
        "dataset": "brats",
        "target": args.target,
        "method": args.method,
        "epsilon": args.epsilon if args.method != "baseline" else None,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "patch_size": args.patch_size,
        "best_val_dice": max(history["val_dice"]),
        "final_val_dice": history["val_dice"][-1],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
    }

    # Sliding window inference on full validation volumes
    if not args.skip_volume_eval:
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
        print(f"\nResults:")
        print(f"  Best Val Dice (patches): {results['best_val_dice']:.4f}")
        print(f"  Full Volume Dice: {results['full_volume_dice']:.4f}")
    else:
        print(f"\nResults:")
        print(f"  Best Val Dice (patches): {results['best_val_dice']:.4f}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

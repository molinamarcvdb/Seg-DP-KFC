#!/usr/bin/env python3
"""DP-SGD training with frozen STU-Net-S encoder for hepatic vessel segmentation.

Methods:
    baseline       - Non-DP (upper bound)
    dp             - DP-SGD on seg_head only (vanilla)
    dp_synthetic   - DP-SGD + AdaDPS preconditioner (synthetic data)
    dp_shampoo_synth - DP-SGD + Shampoo preconditioner (synthetic data)

Usage:
    python scripts/train_hepatic_stunet.py --method baseline --epochs 30
    python scripts/train_hepatic_stunet.py --method dp --epsilon 8.0 --epochs 30
    python scripts/train_hepatic_stunet.py --method dp_shampoo_synth --epsilon 8.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import GradSampleModule
from opacus.accountants.utils import get_noise_multiplier
from tqdm import tqdm

from src.data.hepatic_dataset import (
    HepaticVesselDataset, PreprocessedHepaticDataset,
    HEPATIC_DEFAULT_ROOT, HEPATIC_PREPROCESSED_ROOT,
)
from src.models.stunet_seg import STUNetSeg
from src.training import (
    AdaDPSPreconditioner,
    ShampooPreconditioner,
    DiceBCELoss,
)
from src.training.dp_trainer import clip_and_noise_gradients


ALL_METHODS = ["baseline", "dp", "dp_synthetic", "dp_shampoo_synth"]
DP_METHODS = {"dp", "dp_synthetic", "dp_shampoo_synth"}
SHAMPOO_METHODS = {"dp_shampoo_synth"}
ADADPS_METHODS = {"dp_synthetic"}


def dice_score(pred, target, smooth=1e-6):
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    return ((2.0 * intersection + smooth) / (union + smooth)).item()


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_epoch_dp(model, loader, optimizer, loss_fn,
                   max_grad_norm, noise_multiplier, preconditioner, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training (DP)", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        if loss.dim() > 0:
            loss = loss.mean()
        loss.backward()

        if preconditioner is not None and preconditioner.initialized:
            preconditioner.apply(model)

        clip_and_noise_gradients(model, batch_size, max_grad_norm, noise_multiplier)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(n_batches, 1)


def train_epoch_nodp(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        if loss.dim() > 0:
            loss = loss.mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_masks = []
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, masks)
        if loss.dim() > 0:
            loss = loss.mean()

        probs = torch.sigmoid(outputs)
        all_probs.append(probs.cpu())
        all_masks.append(masks.cpu())
        total_loss += loss.item()
        n_batches += 1

    all_probs = torch.cat(all_probs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    best_dice = 0.0
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        preds = (all_probs > thresh).float()
        intersection = (preds * all_masks).sum()
        union = preds.sum() + all_masks.sum()
        d = ((2.0 * intersection + 1e-6) / (union + 1e-6)).item()
        if d > best_dice:
            best_dice = d

    return total_loss / max(n_batches, 1), best_dice


# ---------------------------------------------------------------------------
# Sliding window inference
# ---------------------------------------------------------------------------

def sliding_window_inference(model, volume, patch_size, device, out_channels=1, batch_size=4):
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


def evaluate_full_volumes(model, dataset, patch_size, device, n_volumes=None):
    n_cases = len(dataset.cases)
    if n_volumes is not None:
        n_cases = min(n_cases, n_volumes)

    all_preds = []
    all_masks = []

    print(f"  Running inference on {n_cases} volumes...")
    for i in range(n_cases):
        volume, mask = dataset.get_full_volume(i)
        pred = sliding_window_inference(model, volume, patch_size, device)
        all_preds.append(pred)
        all_masks.append(mask)

    best_thresh = 0.5
    best_mean_dice = 0.0

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        dices = []
        for pred, mask in zip(all_preds, all_masks):
            pred_binary = (pred > thresh).float()
            p = pred_binary[0]
            m = mask[0]
            intersection = (p * m).sum()
            union = p.sum() + m.sum()
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
            dices.append(dice.item())

        mean_dice = np.mean(dices)
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            best_thresh = thresh

    print(f"  Best threshold: {best_thresh}")

    final_dices = []
    for i, (pred, mask) in enumerate(zip(all_preds, all_masks)):
        pred_binary = (pred > best_thresh).float()
        intersection = (pred_binary[0] * mask[0]).sum()
        union = pred_binary[0].sum() + mask[0].sum()
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        final_dices.append(dice.item())
        print(f"  Volume {i+1}/{n_cases}: Dice={dice.item():.4f}")

    mean_dice = np.mean(final_dices)
    print(f"  Mean Dice: {mean_dice:.4f}")
    return {"mean_dice": mean_dice, "best_threshold": best_thresh}


# ---------------------------------------------------------------------------
# Preconditioner estimation
# ---------------------------------------------------------------------------

class Synthetic3DDataset(torch.utils.data.Dataset):
    """Simple 3D synthetic dataset for preconditioner estimation."""

    def __init__(self, n_samples=500, patch_size=96):
        self.n_samples = n_samples
        self.patch_size = patch_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ps = self.patch_size
        rng = np.random.RandomState(idx)
        # Random noise volume
        image = rng.randn(1, ps, ps, ps).astype(np.float32)
        # Random blob mask
        mask = np.zeros((1, ps, ps, ps), dtype=np.float32)
        n_blobs = rng.randint(2, 6)
        for _ in range(n_blobs):
            c = rng.randint(ps // 4, 3 * ps // 4, size=3)
            r = rng.randint(3, ps // 6)
            z, y, x = np.ogrid[-c[0]:ps-c[0], -c[1]:ps-c[1], -c[2]:ps-c[2]]
            sphere = (z*z + y*y + x*x) <= r*r
            mask[0][sphere] = 1.0
        return torch.from_numpy(image), torch.from_numpy(mask)


def estimate_preconditioner(preconditioner, model, loss_fn, device,
                            patch_size, precond_samples=500, batch_size=2):
    """Estimate preconditioner from 3D synthetic data."""
    print("\nEstimating preconditioner from synthetic 3D data...")

    needs_gsm = isinstance(preconditioner, AdaDPSPreconditioner)
    if needs_gsm:
        est_model = GradSampleModule(model, strict=False).to(device)
    else:
        est_model = model

    synth_ds = Synthetic3DDataset(n_samples=precond_samples, patch_size=patch_size)
    synth_loader = DataLoader(synth_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    num_steps = min(len(synth_loader), precond_samples // batch_size)

    preconditioner.estimate_from_loader(
        model=est_model, data_loader=synth_loader, loss_fn=loss_fn,
        num_steps=num_steps, task="segmentation",
    )

    if needs_gsm:
        est_model.remove_hooks()
        del est_model

    print(f"  Preconditioner estimated ({num_steps} steps)")


def refresh_preconditioner(preconditioner, model, loss_fn, device,
                           patch_size, precond_samples=500, batch_size=2):
    """Re-estimate preconditioner from synthetic data using current model weights."""
    is_gsm = hasattr(model, '_module')
    needs_gsm = isinstance(preconditioner, AdaDPSPreconditioner)

    if needs_gsm:
        est_model = model
    else:
        est_model = model._module if is_gsm else model

    synth_ds = Synthetic3DDataset(n_samples=precond_samples, patch_size=patch_size)
    synth_loader = DataLoader(synth_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    num_steps = min(len(synth_loader), precond_samples // batch_size)

    preconditioner.estimate_from_loader(
        model=est_model, data_loader=synth_loader, loss_fn=loss_fn,
        num_steps=num_steps, task="segmentation",
    )

    print(f"  Preconditioner refreshed ({num_steps} steps)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hepatic STU-Net-S DP-SGD training")
    parser.add_argument("--method", type=str, required=True, choices=ALL_METHODS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--patches_per_volume", type=int, default=16)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--shampoo_damping", type=float, default=1e-4)
    parser.add_argument("--adadps_damping", type=float, default=0.1)
    parser.add_argument("--precond_samples", type=int, default=500)
    parser.add_argument("--refresh_precond", type=int, default=0)
    parser.add_argument("--mid_ch", type=int, default=64,
                        help="Hidden channels in seg head")
    parser.add_argument("--weights_path", type=str, default="pretrained/stunet_small_ep4k.model")
    parser.add_argument("--data_root", type=str, default=HEPATIC_DEFAULT_ROOT)
    parser.add_argument("--preprocessed", type=str, default=None,
                        help="Path to preprocessed .npz dir (skips raw NIfTI loading)")
    parser.add_argument("--target", type=str, default="all",
                        choices=["vessel", "tumour", "all"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_volumes", type=int, default=None)
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of train/val volumes to use (0-1)")
    parser.add_argument("--cache_volumes", type=int, default=8,
                        help="Volumes to cache in RAM (set >= num volumes to preload all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    torch.manual_seed(args.seed)
    start_time = time.time()

    method = args.method
    is_dp = method in DP_METHODS
    needs_shampoo = method in SHAMPOO_METHODS
    needs_adadps = method in ADADPS_METHODS

    print(f"\n{'='*60}")
    print(f"Hepatic STU-Net-S: {method}")
    print(f"  epsilon={args.epsilon}, lr={args.lr}, optimizer={args.optimizer}")
    print(f"  patch_size={args.patch_size}, batch_size={args.batch_size}")
    print(f"{'='*60}")

    # --- Data ---
    if args.preprocessed:
        print(f"\nLoading preprocessed hepatic data from {args.preprocessed}...")
        train_dataset = PreprocessedHepaticDataset(
            root=args.preprocessed, split="train", patch_size=args.patch_size,
            patches_per_volume=args.patches_per_volume, target=args.target,
            augment=True, seed=args.seed, cache_volumes=args.cache_volumes,
        )
        val_dataset = PreprocessedHepaticDataset(
            root=args.preprocessed, split="val", patch_size=args.patch_size,
            patches_per_volume=args.patches_per_volume, target=args.target,
            augment=False, seed=args.seed, cache_volumes=args.cache_volumes,
        )
    else:
        print("\nLoading Hepatic Vessel (Task08) from raw NIfTI...")
        train_dataset = HepaticVesselDataset(
            root=args.data_root, split="train", patch_size=args.patch_size,
            patches_per_volume=args.patches_per_volume, target=args.target,
            augment=True, seed=args.seed, cache_volumes=args.cache_volumes,
        )
        val_dataset = HepaticVesselDataset(
            root=args.data_root, split="val", patch_size=args.patch_size,
            patches_per_volume=args.patches_per_volume, target=args.target,
            augment=False, seed=args.seed, cache_volumes=args.cache_volumes,
        )

    # Subset volumes if data_fraction < 1
    if args.data_fraction < 1.0:
        n_train = max(1, int(len(train_dataset.cases) * args.data_fraction))
        n_val = max(1, int(len(val_dataset.cases) * args.data_fraction))
        train_dataset.cases = train_dataset.cases[:n_train]
        val_dataset.cases = val_dataset.cases[:n_val]
        print(f"  Using {args.data_fraction:.0%} of data: {n_train} train, {n_val} val volumes")

    nw = args.num_workers
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    print(f"  Train: {len(train_dataset)} patches ({len(train_dataset.cases)} volumes)")
    print(f"  Val: {len(val_dataset)} patches ({len(val_dataset.cases)} volumes)")

    # --- Model ---
    model = STUNetSeg(
        weights_path=args.weights_path,
        num_classes=1,
        mid_ch=args.mid_ch,
        device=args.device,
    )

    param_counts = model.count_params()
    print(f"Parameters:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")

    # --- Loss ---
    loss_fn = DiceBCELoss()

    # --- Preconditioner ---
    preconditioner = None
    if needs_shampoo or needs_adadps:
        if needs_shampoo:
            preconditioner = ShampooPreconditioner(
                damping=args.shampoo_damping, device=args.device,
            )
        else:
            preconditioner = AdaDPSPreconditioner(
                damping=args.adadps_damping, device=args.device,
            )

        estimate_preconditioner(
            preconditioner, model, loss_fn, args.device,
            args.patch_size, precond_samples=args.precond_samples,
            batch_size=args.batch_size,
        )

    # --- Optimizer ---
    trainable_params = list(model.get_trainable_params())
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr,
                                    momentum=args.momentum, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # --- DP setup ---
    noise_multiplier = 0.0
    if is_dp:
        model = GradSampleModule(model, strict=False)
        model = model.to(args.device)

        train_size = len(train_loader.dataset)
        sample_rate = args.batch_size / train_size

        noise_multiplier = get_noise_multiplier(
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            sample_rate=sample_rate,
            epochs=args.epochs,
            accountant="rdp",
        )

        print(f"\nDP-SGD Configuration:")
        print(f"  epsilon={args.epsilon}, delta={args.delta}")
        print(f"  noise_multiplier={noise_multiplier:.4f}")
        print(f"  max_grad_norm={args.max_grad_norm}")
        print(f"  sample_rate={sample_rate:.6f}")

    # --- Training ---
    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Refresh preconditioner
        if (preconditioner is not None and args.refresh_precond > 0
                and epoch > 0 and epoch % args.refresh_precond == 0):
            refresh_preconditioner(
                preconditioner, model, loss_fn, args.device,
                args.patch_size, precond_samples=args.precond_samples,
                batch_size=args.batch_size,
            )

        if is_dp:
            train_loss = train_epoch_dp(
                model, train_loader, optimizer, loss_fn,
                args.max_grad_norm, noise_multiplier, preconditioner,
                args.device,
            )
        else:
            train_loss = train_epoch_nodp(
                model, train_loader, optimizer, loss_fn, args.device,
            )

        val_loss, val_dice = validate(model, val_loader, loss_fn, args.device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f} (best: {best_dice:.4f})")

    elapsed = time.time() - start_time

    # --- Sliding window eval ---
    print("\nSliding window inference on validation volumes...")
    model_for_eval = model._module if hasattr(model, '_module') else model
    model_for_eval.eval()
    vol_results = evaluate_full_volumes(
        model_for_eval, val_dataset, args.patch_size, args.device,
        n_volumes=args.eval_volumes,
    )

    # --- Save ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"hepatic_stunet_{method}_eps{args.epsilon}_s{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    results = {
        "dataset": "hepatic",
        "model": "stunet_s",
        "method": method,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "epochs": args.epochs,
        "epsilon": args.epsilon if is_dp else None,
        "seed": args.seed,
        "patch_size": args.patch_size,
        "mid_ch": args.mid_ch,
        "refresh_precond": args.refresh_precond,
        "best_val_dice": best_dice,
        "final_val_dice": history["val_dice"][-1],
        "full_volume_dice": vol_results["mean_dice"],
        "elapsed_seconds": elapsed,
        "param_counts": param_counts,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: {method}")
    print(f"  Best Val Dice (patches): {best_dice:.4f}")
    print(f"  Full Volume Dice: {vol_results['mean_dice']:.4f}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Results: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

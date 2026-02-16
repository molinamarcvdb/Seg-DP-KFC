#!/usr/bin/env python3
"""DP-LoRA finetuning of RADIO backbone for medical image segmentation.

Experiment 6 for MICCAI 2026 paper: DP-Shampoo preconditioning with LoRA.

Methods:
    lora_baseline      - Non-DP LoRA finetuning (upper bound)
    lora_dp            - DP-SGD on LoRA + seg_head (vanilla)
    lora_dp_shampoo    - DP-SGD + Shampoo preconditioner (public data)
    lora_dp_shampoo_synth - DP-SGD + Shampoo (synthetic data)
    lora_dp_adadps     - DP-SGD + AdaDPS diagonal preconditioner
    lora_dp_ffa        - FFA-LoRA (freeze A, DP-SGD on B + seg_head)
    lora_dp_ffa_shampoo - FFA-LoRA + Shampoo

Usage:
    python scripts/train_lora_dp.py --method lora_dp_shampoo --epsilon 8.0
    python scripts/train_lora_dp.py --method lora_baseline --epochs 30
    python scripts/train_lora_dp.py --method lora_dp_ffa --epsilon 4.0
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
from opacus.accountants.utils import get_noise_multiplier
from tqdm import tqdm

from src.data.kvasir_radio_dataset import KvasirRADIODataset, get_kvasir_radio_dataloaders
from src.models.radio_lora import RADIOLoRA
from src.training import (
    AdaDPSPreconditioner,
    ShampooPreconditioner,
    DiceBCELoss,
)
from src.training.preconditioner import clear_grad_samples
from src.training.dp_trainer import compute_per_sample_norms, clip_and_noise_gradients


ALL_METHODS = [
    "lora_baseline",
    "lora_dp",
    "lora_dp_shampoo",
    "lora_dp_shampoo_synth",
    "lora_dp_adadps",
    "lora_dp_ffa",
    "lora_dp_ffa_shampoo",
]

FFA_METHODS = {"lora_dp_ffa", "lora_dp_ffa_shampoo"}
DP_METHODS = {m for m in ALL_METHODS if m != "lora_baseline"}
SHAMPOO_METHODS = {"lora_dp_shampoo", "lora_dp_shampoo_synth", "lora_dp_ffa_shampoo"}
ADADPS_METHODS = {"lora_dp_adadps"}
SYNTH_METHODS = {"lora_dp_shampoo_synth"}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute Dice score from probabilities and binary masks."""
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    return ((2.0 * intersection + smooth) / (union + smooth)).item()


def train_epoch_dp(
    model, train_loader, optimizer, loss_fn,
    max_grad_norm, noise_multiplier, preconditioner, device,
):
    """One training epoch with DP-SGD + optional preconditioning."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc="Training (DP)", leave=False)
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

        # Apply preconditioning BEFORE clipping
        if preconditioner is not None and preconditioner.initialized:
            preconditioner.apply(model)

        # DP-SGD: clip + noise
        clip_and_noise_gradients(model, batch_size, max_grad_norm, noise_multiplier)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(n_batches, 1)


def train_epoch_nodp(model, train_loader, optimizer, loss_fn, device):
    """One training epoch without DP."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
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
def validate(model, val_loader, loss_fn, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_masks = []
    n_batches = 0

    for images, masks in val_loader:
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

    # Find optimal threshold
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
# Preconditioner estimation
# ---------------------------------------------------------------------------

def refresh_preconditioner_from_synthetic(
    preconditioner, model, loss_fn, device, image_size,
    precond_samples=1000, batch_size=8,
):
    """Re-estimate preconditioner from synthetic data using current model weights.

    Unlike initial estimation, this does NOT touch LoRA init — it uses the
    model as-is, capturing the gradient geometry at the current training state.
    Works with GradSampleModule-wrapped models by operating on the inner module.

    For Shampoo: uses param.grad (batch-averaged) on the unwrapped base model.
    For AdaDPS: uses the existing GSM-wrapped model (already has hooks for
    per-sample gradients, so no need to re-wrap).
    """
    from src.data.synthetic import SyntheticSegmentationDataset

    # AdaDPS needs per-sample gradients → use the existing GSM wrapper.
    # Shampoo needs param.grad → use the unwrapped base model.
    needs_gsm = isinstance(preconditioner, AdaDPSPreconditioner)
    is_gsm = hasattr(model, '_module')

    if needs_gsm:
        # Use the existing GSM-wrapped model directly (already has hooks)
        est_model = model
    else:
        # Shampoo: use unwrapped model for batch-averaged param.grad
        est_model = model._module if is_gsm else model

    synth_ds = SyntheticSegmentationDataset(
        n_samples=precond_samples, image_size=image_size,
        in_channels=3, noise_type="pink", mask_strategy="gaussian_blobs",
    )
    synth_loader = DataLoader(synth_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    num_steps = min(len(synth_loader), precond_samples // batch_size)

    preconditioner.estimate_from_loader(
        model=est_model, data_loader=synth_loader, loss_fn=loss_fn,
        num_steps=num_steps, task="segmentation",
    )

    print(f"  Preconditioner refreshed from synthetic data ({num_steps} steps)")


def estimate_preconditioner(
    preconditioner, method, model, loss_fn,
    train_loader, device, image_size, precond_samples=1000, batch_size=8,
):
    """Estimate preconditioner from public or synthetic data.

    For the estimation step we need a model that can do forward/backward
    and produces param.grad for trainable parameters.  We use a fresh
    RADIOLoRA (not wrapped with GradSampleModule) so that standard
    autograd produces .grad tensors.

    Important: LoRA B (lora_up) is initialized to zero, which means A (lora_down)
    gets zero gradients during standard initialization. We temporarily initialize
    B with small random values so the Shampoo/AdaDPS factors capture realistic
    gradient geometry. model.reset_lora() is called after estimation.
    """
    print(f"\nEstimating preconditioner for method={method}...")

    # Temporarily init lora_up (B) with small values so lora_down (A) gets
    # non-zero gradients during estimation. Without this, Shampoo factors for
    # A are just scaled identity (from damping), causing ~100x amplification.
    for _, lora in model.lora_layers:
        nn.init.normal_(lora.lora_up.weight, std=0.01)

    # AdaDPS needs per-sample gradients (grad_sample) for estimation,
    # so we temporarily wrap with GradSampleModule.
    # Shampoo uses batch-averaged param.grad and doesn't need wrapping.
    needs_gsm = isinstance(preconditioner, AdaDPSPreconditioner)
    if needs_gsm:
        model_for_est = GradSampleModule(model, strict=False)
        model_for_est = model_for_est.to(device)
    else:
        model_for_est = model

    if method in SYNTH_METHODS:
        from src.data.synthetic import SyntheticSegmentationDataset
        synth_ds = SyntheticSegmentationDataset(
            n_samples=precond_samples,
            image_size=image_size,
            in_channels=3,
            noise_type="pink",
            mask_strategy="gaussian_blobs",
        )
        synth_loader = DataLoader(synth_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        num_steps = min(len(synth_loader), precond_samples // batch_size)
        preconditioner.estimate_from_loader(
            model=model_for_est, data_loader=synth_loader, loss_fn=loss_fn,
            num_steps=num_steps, task="segmentation",
        )
    else:
        # Public data estimation: use the training data (oracle setting)
        num_steps = min(100, len(train_loader))
        preconditioner.estimate_from_loader(
            model=model_for_est, data_loader=train_loader, loss_fn=loss_fn,
            num_steps=num_steps, task="segmentation",
        )

    # Properly remove GSM hooks so the model can be re-wrapped for training
    if needs_gsm:
        model_for_est.remove_hooks()
        del model_for_est


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DP-LoRA finetuning with RADIO")
    parser.add_argument("--method", type=str, required=True, choices=ALL_METHODS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=352)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--num_lora_layers", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"],
                        help="Optimizer: adamw (default) or sgd (avoids double-preconditioning)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum (only used with --optimizer sgd)")
    parser.add_argument("--shampoo_damping", type=float, default=1e-4)
    parser.add_argument("--adadps_damping", type=float, default=0.1)
    parser.add_argument("--precond_samples", type=int, default=1000)
    parser.add_argument("--refresh_precond", type=int, default=0,
                        help="Re-estimate preconditioner every N epochs from synthetic data (0=disabled). "
                             "Free privacy-wise since it uses synthetic data only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default="./data/kvasir_seg")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    torch.manual_seed(args.seed)
    start_time = time.time()

    method = args.method
    is_dp = method in DP_METHODS
    is_ffa = method in FFA_METHODS
    needs_shampoo = method in SHAMPOO_METHODS
    needs_adadps = method in ADADPS_METHODS

    print(f"\n{'='*60}")
    print(f"DP-LoRA Finetuning: {method}")
    print(f"  epsilon={args.epsilon}, lr={args.lr}, rank={args.lora_rank}")
    print(f"  optimizer={args.optimizer}, ffa_mode={is_ffa}, dp={is_dp}")
    print(f"{'='*60}")

    # --- Data ---
    train_loader, val_loader = get_kvasir_radio_dataloaders(
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")

    # --- Model ---
    model = RADIOLoRA(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_lora_layers=args.num_lora_layers,
        num_classes=1,
        device=args.device,
        ffa_mode=is_ffa,
    )

    param_counts = model.count_params()
    print(f"Parameters:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")

    # --- Loss ---
    loss_fn = DiceBCELoss()

    # --- Preconditioner estimation (before model reset) ---
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

        # Estimate using unwrapped model (uses param.grad, not grad_sample)
        estimate_preconditioner(
            preconditioner, method, model, loss_fn,
            train_loader, args.device, args.image_size,
            precond_samples=args.precond_samples, batch_size=args.batch_size,
        )

        # Reset LoRA after estimation so training starts fresh
        model.reset_lora()

    # --- Optimizer (only trainable params) ---
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
        # Wrap model with GradSampleModule for per-sample gradients
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
        print(f"  sample_rate={sample_rate:.4f}")

    # --- Training ---
    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Refresh preconditioner periodically (uses synthetic data, free privacy)
        if (preconditioner is not None and args.refresh_precond > 0
                and epoch > 0 and epoch % args.refresh_precond == 0):
            refresh_preconditioner_from_synthetic(
                preconditioner, model, loss_fn, args.device, args.image_size,
                precond_samples=args.precond_samples, batch_size=args.batch_size,
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

    # --- Save results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"lora_{method}_eps{args.epsilon}_s{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    results = {
        "dataset": "kvasir",
        "method": method,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "epochs": args.epochs,
        "epsilon": args.epsilon if is_dp else None,
        "seed": args.seed,
        "lora_rank": args.lora_rank,
        "ffa_mode": is_ffa,
        "refresh_precond": args.refresh_precond,
        "best_val_dice": best_dice,
        "final_val_dice": history["val_dice"][-1],
        "elapsed_seconds": elapsed,
        "param_counts": param_counts,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: {method}")
    print(f"  Best Val Dice: {best_dice:.4f}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Results: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

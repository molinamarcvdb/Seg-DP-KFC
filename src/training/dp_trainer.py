"""
DP-SGD Trainer for Segmentation with Preconditioning.

Implements differentially private training using Opacus with optional
AdaDPS preconditioning for improved convergence.
"""

from typing import Dict, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import GradSampleModule
from opacus.accountants.utils import get_noise_multiplier
from tqdm import tqdm

from .preconditioner import AdaDPSPreconditioner, create_preconditioner, clear_grad_samples


def compute_per_sample_norms(model: nn.Module, batch_size: int) -> torch.Tensor:
    """Compute L2 norm of per-sample gradients."""
    base_model = model._module if hasattr(model, '_module') else model
    device = next(base_model.parameters()).device

    norms_sq = torch.zeros(batch_size, device=device)

    for param in base_model.parameters():
        gs = getattr(param, 'grad_sample', None)
        if gs is None:
            continue

        # Handle list format
        if isinstance(gs, list):
            gs = torch.stack(gs, dim=0) if gs[0].shape == param.shape else torch.cat(gs, dim=0)

        # Flatten and compute squared norm
        flat = gs.reshape(batch_size, -1)
        norms_sq += flat.norm(2, dim=1) ** 2

    return norms_sq.sqrt()


def clip_and_noise_gradients(
    model: nn.Module,
    batch_size: int,
    max_grad_norm: float,
    noise_multiplier: float,
) -> None:
    """
    Apply DP-SGD: clip per-sample gradients and add noise.

    1. Compute per-sample gradient norms
    2. Clip gradients that exceed max_grad_norm
    3. Sum clipped gradients
    4. Add Gaussian noise calibrated to (noise_multiplier * max_grad_norm)
    5. Average to get final gradient
    """
    base_model = model._module if hasattr(model, '_module') else model
    device = next(base_model.parameters()).device

    # Compute per-sample norms
    norms = compute_per_sample_norms(model, batch_size)

    # Compute clipping factor: min(1, C/||g||)
    clip_factor = (max_grad_norm / (norms + 1e-6)).clamp(max=1.0)

    # Clip, sum, add noise, and average
    for param in base_model.parameters():
        gs = getattr(param, 'grad_sample', None)
        if gs is None:
            continue

        # Handle list format
        if isinstance(gs, list):
            gs = torch.stack(gs, dim=0) if gs[0].shape == param.shape else torch.cat(gs, dim=0)

        # Clip each sample's gradient
        flat = gs.reshape(batch_size, -1)
        clipped = flat * clip_factor.unsqueeze(1)

        # Sum clipped gradients
        summed = clipped.sum(dim=0)

        # Add noise: N(0, (σ * C)² * I)
        noise_std = noise_multiplier * max_grad_norm
        noise = torch.randn_like(summed) * noise_std

        # Average to get final gradient
        param.grad = ((summed + noise) / batch_size).view_as(param)

        # Clear grad_sample
        param.grad_sample = None


class DPTrainer:
    """
    Differentially Private Trainer with optional preconditioning.

    Supports:
    - DP-SGD with Opacus GradSampleModule
    - AdaDPS preconditioning (estimate from public data)
    - Segmentation losses (BCE, Dice, combined)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        epsilon: float,
        delta: float,
        max_grad_norm: float,
        epochs: int,
        device: str = "cuda",
        preconditioner: Optional[AdaDPSPreconditioner] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Args:
            model: Model (will be wrapped with GradSampleModule if not already)
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            loss_fn: Loss function
            epsilon: Privacy budget
            delta: Privacy parameter
            max_grad_norm: Gradient clipping norm
            epochs: Number of training epochs
            device: Device to train on
            preconditioner: Optional preconditioner for gradients
            scheduler: Optional learning rate scheduler
        """
        self.device = device
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.preconditioner = preconditioner
        self.scheduler = scheduler
        self.loss_fn = loss_fn

        # Use model as-is if already wrapped, otherwise wrap it
        if isinstance(model, GradSampleModule):
            self.model = model
        else:
            self.model = GradSampleModule(model)
        self.model = self.model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer

        # Compute noise multiplier for target privacy
        train_size = len(train_loader.dataset)
        sample_rate = train_loader.batch_size / train_size

        self.noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="rdp",
        )

        print(f"DP-SGD Configuration:")
        print(f"  ε={epsilon}, δ={delta}")
        print(f"  Noise multiplier: {self.noise_multiplier:.4f}")
        print(f"  Max grad norm: {max_grad_norm}")
        print(f"  Sample rate: {sample_rate:.4f}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with DP-SGD."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
            else:
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)

            batch_size = images.size(0)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)

            if loss.dim() > 0:
                loss = loss.mean()

            # Backward pass (creates grad_sample)
            loss.backward()

            # Apply preconditioning BEFORE clipping (if enabled)
            if self.preconditioner is not None and self.preconditioner.initialized:
                self.preconditioner.apply(self.model)

            # DP-SGD: clip and add noise
            clip_and_noise_gradients(
                self.model,
                batch_size,
                self.max_grad_norm,
                self.noise_multiplier,
            )

            # Optimizer step
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {"train_loss": total_loss / n_batches}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluate on validation set with optimal threshold selection."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # Collect all predictions and targets for optimal threshold search
        all_probs = []
        all_masks = []
        all_fovs = []

        for batch in self.val_loader:
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                fov = batch.get("fov", torch.ones_like(masks)).to(self.device)
            else:
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)
                fov = torch.ones_like(masks)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            if loss.dim() > 0:
                loss = loss.mean()

            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu())
            all_masks.append(masks.cpu())
            all_fovs.append(fov.cpu())

            total_loss += loss.item()
            n_batches += 1

        # Concatenate all batches
        all_probs = torch.cat(all_probs, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        all_fovs = torch.cat(all_fovs, dim=0)

        # Find optimal threshold
        best_dice = 0.0
        best_thresh = 0.5
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            preds = (all_probs > thresh).float()
            dice = self._dice_score(preds, all_masks, all_fovs).item()
            if dice > best_dice:
                best_dice = dice
                best_thresh = thresh

        return {
            "val_loss": total_loss / n_batches,
            "val_dice": best_dice,
            "best_threshold": best_thresh,
        }

    def _dice_score(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fov: Optional[torch.Tensor] = None,
        smooth: float = 1e-6,
    ) -> torch.Tensor:
        """Compute Dice score, optionally masked by FOV."""
        if fov is not None:
            pred = pred * fov
            target = target * fov

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        return (2.0 * intersection + smooth) / (union + smooth)

    def fit(
        self,
        log_fn: Optional[Callable[[int, Dict], None]] = None,
        save_fn: Optional[Callable[[int, nn.Module], None]] = None,
    ) -> Dict[str, list]:
        """
        Train for all epochs.

        Args:
            log_fn: Optional callback for logging (epoch, metrics)
            save_fn: Optional callback for saving checkpoints (epoch, model)

        Returns:
            History dict with train/val metrics per epoch
        """
        history = {"train_loss": [], "val_loss": [], "val_dice": []}

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_dice"].append(val_metrics["val_dice"])

            # Log
            metrics = {**train_metrics, **val_metrics}
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Dice: {val_metrics['val_dice']:.4f}")

            if log_fn:
                log_fn(epoch, metrics)

            if save_fn:
                save_fn(epoch, self.model)

        return history


class NonDPTrainer:
    """
    Standard (non-private) trainer for comparison/pretraining.

    Same interface as DPTrainer but without DP-SGD.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        epochs: int,
        device: str = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device
        self.scheduler = scheduler

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch (standard, no DP)."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
            else:
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)

            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {"train_loss": total_loss / n_batches}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluate on validation set with optimal threshold selection."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # Collect all predictions and targets for optimal threshold search
        all_probs = []
        all_masks = []
        all_fovs = []

        for batch in self.val_loader:
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                fov = batch.get("fov", torch.ones_like(masks)).to(self.device)
            else:
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)
                fov = torch.ones_like(masks)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            if loss.dim() > 0:
                loss = loss.mean()

            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu())
            all_masks.append(masks.cpu())
            all_fovs.append(fov.cpu())

            total_loss += loss.item()
            n_batches += 1

        # Concatenate all batches
        all_probs = torch.cat(all_probs, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        all_fovs = torch.cat(all_fovs, dim=0)

        # Find optimal threshold
        best_dice = 0.0
        best_thresh = 0.5
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            preds = (all_probs > thresh).float()
            dice = self._dice_score(preds, all_masks, all_fovs).item()
            if dice > best_dice:
                best_dice = dice
                best_thresh = thresh

        return {
            "val_loss": total_loss / n_batches,
            "val_dice": best_dice,
            "best_threshold": best_thresh,
        }

    def _dice_score(self, pred, target, fov=None, smooth=1e-6):
        if fov is not None:
            pred = pred * fov
            target = target * fov
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2.0 * intersection + smooth) / (union + smooth)

    def fit(self, log_fn=None, save_fn=None):
        history = {"train_loss": [], "val_loss": [], "val_dice": []}

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if self.scheduler:
                self.scheduler.step()

            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_dice"].append(val_metrics["val_dice"])

            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Dice: {val_metrics['val_dice']:.4f}")

            if log_fn:
                log_fn(epoch, {**train_metrics, **val_metrics})
            if save_fn:
                save_fn(epoch, self.model)

        return history


# =============================================================================
# Classification Trainers
# =============================================================================

class DPClassificationTrainer:
    """
    Differentially Private Trainer for Classification with optional preconditioning.

    Supports:
    - DP-SGD with Opacus GradSampleModule
    - AdaDPS preconditioning (from public or synthetic data)
    - Multi-class classification
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        epsilon: float,
        delta: float,
        max_grad_norm: float,
        epochs: int,
        device: str = "cuda",
        preconditioner: Optional[AdaDPSPreconditioner] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Args:
            model: Model (will be wrapped with GradSampleModule if not already)
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            loss_fn: Loss function (e.g., CrossEntropyLoss)
            epsilon: Privacy budget
            delta: Privacy parameter
            max_grad_norm: Gradient clipping norm
            epochs: Number of training epochs
            device: Device to train on
            preconditioner: Optional preconditioner for gradients
            scheduler: Optional learning rate scheduler
        """
        self.device = device
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.preconditioner = preconditioner
        self.scheduler = scheduler
        self.loss_fn = loss_fn

        # Use model as-is if already wrapped, otherwise wrap it
        if isinstance(model, GradSampleModule):
            self.model = model
        else:
            self.model = GradSampleModule(model)
        self.model = self.model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer

        # Compute noise multiplier for target privacy
        train_size = len(train_loader.dataset)
        sample_rate = train_loader.batch_size / train_size

        self.noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="rdp",
        )

        print(f"DP-SGD Configuration:")
        print(f"  ε={epsilon}, δ={delta}")
        print(f"  Noise multiplier: {self.noise_multiplier:.4f}")
        print(f"  Max grad norm: {max_grad_norm}")
        print(f"  Sample rate: {sample_rate:.4f}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with DP-SGD."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            batch_size = images.size(0)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            if loss.dim() > 0:
                loss = loss.mean()

            # Backward pass (creates grad_sample)
            loss.backward()

            # Apply preconditioning BEFORE clipping (if enabled)
            if self.preconditioner is not None and self.preconditioner.initialized:
                self.preconditioner.apply(self.model)

            # DP-SGD: clip and add noise
            clip_and_noise_gradients(
                self.model,
                batch_size,
                self.max_grad_norm,
                self.noise_multiplier,
            )

            # Optimizer step
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {
            "train_loss": total_loss / len(self.train_loader),
            "train_acc": 100.0 * correct / total,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.val_loader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            if loss.dim() > 0:
                loss = loss.mean()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            "val_loss": total_loss / len(self.val_loader),
            "val_acc": 100.0 * correct / total,
        }

    def fit(
        self,
        log_fn: Optional[Callable[[int, Dict], None]] = None,
        save_fn: Optional[Callable[[int, nn.Module], None]] = None,
    ) -> Dict[str, list]:
        """Train for all epochs."""
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            history["train_loss"].append(train_metrics["train_loss"])
            history["train_acc"].append(train_metrics["train_acc"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_acc"].append(val_metrics["val_acc"])

            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_acc']:.2f}%")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.2f}%")

            if log_fn:
                log_fn(epoch, {**train_metrics, **val_metrics})
            if save_fn:
                save_fn(epoch, self.model)

        return history


class NonDPClassificationTrainer:
    """
    Standard (non-private) trainer for classification.

    Same interface as DPClassificationTrainer but without DP-SGD.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        epochs: int,
        device: str = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device
        self.scheduler = scheduler

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch (standard, no DP)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {
            "train_loss": total_loss / len(self.train_loader),
            "train_acc": 100.0 * correct / total,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.val_loader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            if loss.dim() > 0:
                loss = loss.mean()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return {
            "val_loss": total_loss / len(self.val_loader),
            "val_acc": 100.0 * correct / total,
        }

    def fit(self, log_fn=None, save_fn=None):
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if self.scheduler:
                self.scheduler.step()

            history["train_loss"].append(train_metrics["train_loss"])
            history["train_acc"].append(train_metrics["train_acc"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_acc"].append(val_metrics["val_acc"])

            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_acc']:.2f}%")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.2f}%")

            if log_fn:
                log_fn(epoch, {**train_metrics, **val_metrics})
            if save_fn:
                save_fn(epoch, self.model)

        return history

"""
AdaDPS Diagonal Preconditioning for DP-SGD.

Based on "Private Adaptive Optimization with Side Information" (ICML'22).

Key idea: Use auxiliary (public) data to estimate diagonal preconditioner E[g²],
then apply preconditioning BEFORE gradient clipping in DP-SGD.

For segmentation:
- Public dataset (e.g., DRIVE) provides geometry of gradient space
- Private dataset (e.g., STARE) is trained with DP guarantees
- Preconditioning helps DP-SGD converge faster by normalizing gradient scales
"""

from typing import Dict, Optional, Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def clear_grad_samples(model: nn.Module) -> None:
    """Clear grad_sample from all parameters to free memory."""
    target = model._module if hasattr(model, '_module') else model
    for param in target.parameters():
        if hasattr(param, 'grad_sample'):
            param.grad_sample = None


def get_grad_sample_tensor(param: nn.Parameter) -> Optional[torch.Tensor]:
    """
    Convert grad_sample to tensor format, handling Opacus list case.

    Opacus may store grad_sample as a list of tensors (for hooks) or
    a single tensor. This function normalizes both cases.
    """
    gs = getattr(param, 'grad_sample', None)
    if gs is None:
        return None

    if isinstance(gs, list):
        if len(gs) == 0:
            return None
        first = gs[0]
        # Stack if each element is per-sample, concat if batched
        if first.shape == param.shape:
            return torch.stack(gs, dim=0)
        else:
            return torch.cat(gs, dim=0)
    return gs


class AdaDPSPreconditioner:
    """
    AdaDPS-style diagonal preconditioner for DP-SGD.

    Estimates E[g²] (element-wise second moment of gradients) from auxiliary data,
    then uses 1/sqrt(E[g²] + eps) to precondition per-sample gradients before clipping.

    This helps because:
    1. Different parameters have vastly different gradient scales
    2. Clipping with a fixed norm disproportionately affects large-gradient params
    3. Preconditioning normalizes scales, making clipping more uniform
    """

    def __init__(
        self,
        damping: float = 1e-4,
        device: str = "cuda",
    ):
        """
        Args:
            damping: Added to E[g²] for numerical stability (default 1e-4)
            device: Device for preconditioner tensors
        """
        self.damping = damping
        self.device = device
        self.preconditioner: Dict[str, torch.Tensor] = {}
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def estimate_from_loader(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        num_steps: int,
        show_progress: bool = True,
        task: str = "segmentation",
    ) -> None:
        """
        Estimate preconditioner E[g²] from a data loader.

        Supports both segmentation (images + masks) and classification (images + labels).

        Args:
            model: GradSampleModule-wrapped model
            data_loader: DataLoader for auxiliary (public/synthetic) data
            loss_fn: Loss function (should return per-sample or mean loss)
            num_steps: Number of batches to use for estimation
            show_progress: Show progress bar
            task: "segmentation" or "classification"
        """
        base_model = model._module if hasattr(model, '_module') else model
        model.train()

        # Initialize accumulators
        self.preconditioner = {}
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                self.preconditioner[name] = torch.zeros_like(param, device=self.device)

        # Clear any existing grad_samples
        clear_grad_samples(model)

        data_iter = iter(data_loader)
        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Estimating preconditioner", leave=False)

        for _ in iterator:
            # Get batch (cycle if needed)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                targets = batch.get("mask", batch.get("label")).to(self.device)
            else:
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)

            # Forward pass
            model.zero_grad()
            outputs = model(images)

            # Compute loss (ensure it triggers per-sample gradients)
            loss = loss_fn(outputs, targets)
            if loss.dim() > 0:
                loss = loss.mean()

            # Backward pass (creates grad_sample for each parameter)
            loss.backward()

            # Accumulate E[g²] from per-sample gradients
            for name, param in base_model.named_parameters():
                if not param.requires_grad:
                    continue

                gs = get_grad_sample_tensor(param)
                if gs is None:
                    continue

                # CRITICAL: Square individual gradients THEN average
                # E[g²] = (1/B) * sum(g_i²), NOT (mean(g))²
                batch_second_moment = (gs ** 2).mean(dim=0)
                self.preconditioner[name] += batch_second_moment

            # Clear grad_samples to prevent memory buildup
            clear_grad_samples(model)

        # Average over batches
        for name in self.preconditioner:
            self.preconditioner[name] /= num_steps

        model.zero_grad()
        clear_grad_samples(model)
        self._initialized = True

    def estimate_from_synthetic(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        num_samples: int = 500,
        batch_size: int = 32,
        image_size: int = 28,
        in_channels: int = 3,
        num_classes: int = 9,
        noise_type: str = "pink",
        task: str = "classification",
        show_progress: bool = True,
    ) -> None:
        """
        Estimate preconditioner E[g²] from synthetic data.

        This method generates synthetic data on-the-fly to estimate
        the preconditioner without requiring any real data.

        Args:
            model: GradSampleModule-wrapped model
            loss_fn: Loss function
            num_samples: Number of synthetic samples to use
            batch_size: Batch size for estimation
            image_size: Image size (square)
            in_channels: Number of input channels
            num_classes: Number of classes (for classification)
            noise_type: Type of noise ("white", "pink", "brown", "perlin")
            task: "classification" or "segmentation"
            show_progress: Show progress bar
        """
        # Import here to avoid circular imports
        from ..data.synthetic import (
            SyntheticClassificationDataset,
            SyntheticSegmentationDataset,
        )

        # Create synthetic dataset
        if task == "classification":
            dataset = SyntheticClassificationDataset(
                num_classes=num_classes,
                n_samples=num_samples,
                image_size=image_size,
                in_channels=in_channels,
                noise_type=noise_type,
            )
        else:
            dataset = SyntheticSegmentationDataset(
                n_samples=num_samples,
                image_size=image_size,
                in_channels=in_channels,
                noise_type=noise_type,
            )

        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Use existing method
        num_steps = min(len(data_loader), num_samples // batch_size)
        self.estimate_from_loader(
            model=model,
            data_loader=data_loader,
            loss_fn=loss_fn,
            num_steps=num_steps,
            show_progress=show_progress,
            task=task,
        )

    def apply(self, model: nn.Module) -> None:
        """
        Apply preconditioning to per-sample gradients in-place.

        Transforms: g_precond = g / sqrt(E[g²] + eps)

        Must be called AFTER backward() and BEFORE gradient clipping.
        """
        if not self._initialized:
            raise RuntimeError("Preconditioner not initialized. Call estimate_from_loader first.")

        base_model = model._module if hasattr(model, '_module') else model

        for name, param in base_model.named_parameters():
            gs = getattr(param, 'grad_sample', None)
            if gs is None or name not in self.preconditioner:
                continue

            # Compute scaling factor: 1 / sqrt(E[g²] + eps)
            scale = 1.0 / torch.sqrt(self.preconditioner[name] + self.damping)

            # Apply to per-sample gradients
            if isinstance(gs, list):
                # Handle list format from Opacus hooks
                param.grad_sample = [g * scale.unsqueeze(0) for g in gs]
            else:
                # Tensor format: [batch_size, *param_shape]
                param.grad_sample = gs * scale.unsqueeze(0)

    def save(self, path: str) -> None:
        """Save preconditioner to file."""
        torch.save({
            'preconditioner': self.preconditioner,
            'damping': self.damping,
            'initialized': self._initialized,
        }, path)

    def load(self, path: str) -> None:
        """Load preconditioner from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.preconditioner = checkpoint['preconditioner']
        self.damping = checkpoint['damping']
        self._initialized = checkpoint['initialized']


class IdentityPreconditioner:
    """No-op preconditioner (baseline)."""

    def __init__(self, **kwargs):
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return True

    def estimate_from_loader(self, *args, **kwargs) -> None:
        pass

    def apply(self, model: nn.Module) -> None:
        pass

    def save(self, path: str) -> None:
        torch.save({'type': 'identity'}, path)

    def load(self, path: str) -> None:
        pass


def create_preconditioner(
    precond_type: str,
    damping: float = 1e-4,
    device: str = "cuda",
) -> AdaDPSPreconditioner:
    """
    Factory function to create preconditioner.

    Args:
        precond_type: "none", "adadps_public", "adadps_private", "adadps_synthetic"
        damping: Damping factor for AdaDPS
        device: Device for tensors

    Returns:
        Preconditioner instance
    """
    if precond_type == "none":
        return IdentityPreconditioner()
    elif precond_type in ["adadps_public", "adadps_private", "adadps_synthetic"]:
        return AdaDPSPreconditioner(damping=damping, device=device)
    else:
        raise ValueError(f"Unknown preconditioner type: {precond_type}")

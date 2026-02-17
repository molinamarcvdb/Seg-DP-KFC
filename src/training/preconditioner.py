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
        damping: float = 0.1,
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
        mask_type: str = "blob",
        mask_strategy: str = None,
        task: str = "classification",
        spatial_dims: int = 2,
        show_progress: bool = True,
    ) -> None:
        """
        Estimate preconditioner E[g²] from synthetic data.

        This method generates synthetic data on-the-fly to estimate
        the preconditioner without requiring any real data.
        Automatically selects 2D or 3D generators based on spatial_dims.

        Args:
            model: GradSampleModule-wrapped model
            loss_fn: Loss function
            num_samples: Number of synthetic samples to use
            batch_size: Batch size for estimation
            image_size: Image size (square for 2D, cube for 3D)
            in_channels: Number of input channels
            num_classes: Number of classes (for classification)
            noise_type: Type of noise ("white", "pink", "brown", "perlin")
            mask_type: Legacy mask type for segmentation ("blob", "random", "perlin")
            mask_strategy: Mask strategy name (overrides mask_type if set).
                          Options: "gaussian_blobs", "random_shapes", "pink_threshold",
                          "voronoi", "frangi"
            task: "classification" or "segmentation"
            spatial_dims: 2 for 2D data, 3 for 3D data (auto-selects generators)
            show_progress: Show progress bar
        """
        if task == "classification":
            from ..data.synthetic import SyntheticClassificationDataset
            dataset = SyntheticClassificationDataset(
                num_classes=num_classes,
                n_samples=num_samples,
                image_size=image_size,
                in_channels=in_channels,
                noise_type=noise_type,
            )
        elif spatial_dims == 3:
            from ..data.synthetic3d import SyntheticSegmentation3DDataset
            # Resolve mask strategy
            if mask_strategy is None:
                mask_strategy_map = {
                    "random": "random_shapes",
                    "blob": "gaussian_blobs",
                    "perlin": "pink_threshold",
                }
                mask_strategy = mask_strategy_map.get(mask_type, "gaussian_blobs")
            dataset = SyntheticSegmentation3DDataset(
                n_samples=num_samples,
                patch_size=image_size,
                in_channels=in_channels,
                num_classes=num_classes,
                noise_type=noise_type,
                mask_strategy=mask_strategy,
            )
        else:
            from ..data.synthetic import SyntheticSegmentationDataset
            # Resolve mask strategy
            if mask_strategy is None:
                mask_strategy_map = {
                    "random": "random_shapes",
                    "blob": "gaussian_blobs",
                    "perlin": "pink_threshold",
                }
                mask_strategy = mask_strategy_map.get(mask_type, "gaussian_blobs")
            dataset = SyntheticSegmentationDataset(
                n_samples=num_samples,
                image_size=image_size,
                in_channels=in_channels,
                num_classes=num_classes,
                noise_type=noise_type,
                mask_strategy=mask_strategy,
            )

        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
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


class MomentumPreconditioner:
    """
    Adam-style momentum preconditioner for DP-SGD.

    Improvements over basic AdaDPS:
    1. Exponential moving average of second moments (like Adam's v_t)
    2. Optional first moment tracking (like Adam's m_t)
    3. Bias correction for early iterations
    4. Layer-wise adaptive damping

    This provides more stable gradient geometry estimation and better
    handles varying gradient magnitudes across layers.
    """

    def __init__(
        self,
        damping: float = 0.1,
        beta1: float = 0.0,
        beta2: float = 0.9,
        adaptive_damping: bool = True,
        device: str = "cuda",
    ):
        """
        Args:
            damping: Base damping factor (will be adapted per-layer if adaptive_damping=True)
            beta1: First moment decay (set to 0 to disable, default 0 = second-moment only)
            beta2: Second moment decay. Use 0.9 for few-step estimation (not 0.999 like Adam,
                   since we only run ~15 estimation steps, not thousands of training steps)
            adaptive_damping: Scale damping per-layer based on gradient magnitude
            device: Device for tensors
        """
        self.base_damping = damping
        self.beta1 = beta1
        self.beta2 = beta2
        self.adaptive_damping = adaptive_damping
        self.device = device

        # Moment accumulators
        self.first_moment: Dict[str, torch.Tensor] = {}  # m_t
        self.second_moment: Dict[str, torch.Tensor] = {}  # v_t
        self.damping_per_layer: Dict[str, float] = {}

        self._step = 0
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
        Estimate preconditioner with momentum from a data loader.

        Uses exponential moving average for stable estimation.
        """
        base_model = model._module if hasattr(model, '_module') else model
        model.train()

        # Initialize moments
        self.first_moment = {}
        self.second_moment = {}
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                self.first_moment[name] = torch.zeros_like(param, device=self.device)
                self.second_moment[name] = torch.zeros_like(param, device=self.device)

        clear_grad_samples(model)

        data_iter = iter(data_loader)
        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Estimating preconditioner (momentum)", leave=False)

        for step in iterator:
            self._step = step + 1

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            # Handle batch formats
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                targets = batch.get("mask", batch.get("label")).to(self.device)
            else:
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)

            # Forward + backward
            model.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()

            # Update moments with EMA
            for name, param in base_model.named_parameters():
                if not param.requires_grad:
                    continue

                gs = get_grad_sample_tensor(param)
                if gs is None:
                    continue

                # Compute batch statistics
                batch_mean = gs.mean(dim=0)  # E[g]
                batch_sq_mean = (gs ** 2).mean(dim=0)  # E[g²]

                # Update first moment: m_t = β₁ * m_{t-1} + (1-β₁) * g
                if self.beta1 > 0:
                    self.first_moment[name] = (
                        self.beta1 * self.first_moment[name] +
                        (1 - self.beta1) * batch_mean
                    )

                # Update second moment: v_t = β₂ * v_{t-1} + (1-β₂) * g²
                self.second_moment[name] = (
                    self.beta2 * self.second_moment[name] +
                    (1 - self.beta2) * batch_sq_mean
                )

            clear_grad_samples(model)

        # Compute layer-wise adaptive damping
        if self.adaptive_damping:
            self._compute_adaptive_damping()

        model.zero_grad()
        clear_grad_samples(model)
        self._initialized = True

    def _compute_adaptive_damping(self) -> None:
        """Compute per-layer damping based on gradient magnitudes."""
        bias_correction = 1 - self.beta2 ** self._step
        # With beta2=0.9 and 15 steps: 1 - 0.9^15 ≈ 0.79, much more stable
        bias_correction = max(bias_correction, 1e-3)

        # Get global median of second moments
        all_medians = []
        for name, v in self.second_moment.items():
            v_corrected = v / bias_correction
            layer_median = v_corrected.median().item()
            all_medians.append((name, layer_median))

        median_values = [m for _, m in all_medians]
        global_median = sorted(median_values)[len(median_values) // 2] if median_values else 1.0
        global_median = max(global_median, 1e-10)

        # Scale damping per layer — wider range to avoid over-constraining
        for name, layer_median in all_medians:
            layer_median = max(layer_median, 1e-10)
            ratio = global_median / layer_median
            self.damping_per_layer[name] = self.base_damping * ratio
            # Wider range: floor at base_damping/10 to prevent extreme amplification
            floor = self.base_damping * 0.1
            ceiling = self.base_damping * 10.0
            self.damping_per_layer[name] = max(floor, min(ceiling, self.damping_per_layer[name]))

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
        mask_type: str = "random",
        mask_strategy: str = None,
        task: str = "classification",
        spatial_dims: int = 2,
        show_progress: bool = True,
    ) -> None:
        """Estimate preconditioner from synthetic data with momentum."""
        if task == "classification":
            from ..data.synthetic import SyntheticClassificationDataset
            dataset = SyntheticClassificationDataset(
                num_classes=num_classes,
                n_samples=num_samples,
                image_size=image_size,
                in_channels=in_channels,
                noise_type=noise_type,
            )
        elif spatial_dims == 3:
            from ..data.synthetic3d import SyntheticSegmentation3DDataset
            if mask_strategy is None:
                mask_strategy_map = {"random": "random_shapes", "blob": "gaussian_blobs", "perlin": "pink_threshold"}
                mask_strategy = mask_strategy_map.get(mask_type, "gaussian_blobs")
            dataset = SyntheticSegmentation3DDataset(
                n_samples=num_samples, patch_size=image_size, in_channels=in_channels,
                num_classes=num_classes, noise_type=noise_type, mask_strategy=mask_strategy,
            )
        else:
            from ..data.synthetic import SyntheticSegmentationDataset
            if mask_strategy is None:
                mask_strategy_map = {"random": "random_shapes", "blob": "gaussian_blobs", "perlin": "pink_threshold"}
                mask_strategy = mask_strategy_map.get(mask_type, "gaussian_blobs")
            dataset = SyntheticSegmentationDataset(
                n_samples=num_samples, image_size=image_size, in_channels=in_channels,
                num_classes=num_classes, noise_type=noise_type, mask_strategy=mask_strategy,
            )

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

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
        Apply momentum-based preconditioning to per-sample gradients.

        Uses bias-corrected second moment: v_hat = v / (1 - β₂^t)
        Scale: 1 / sqrt(v_hat + λ)
        """
        if not self._initialized:
            raise RuntimeError("Preconditioner not initialized.")

        base_model = model._module if hasattr(model, '_module') else model

        bias_correction = 1 - self.beta2 ** self._step
        bias_correction = max(bias_correction, 1e-3)

        for name, param in base_model.named_parameters():
            gs = getattr(param, 'grad_sample', None)
            if gs is None or name not in self.second_moment:
                continue

            v_corrected = self.second_moment[name] / bias_correction

            # Get damping (adaptive or base)
            damping = self.damping_per_layer.get(name, self.base_damping)

            # Compute scale: 1 / sqrt(v_hat + λ)
            scale = 1.0 / torch.sqrt(v_corrected + damping)

            # Apply to per-sample gradients
            if isinstance(gs, list):
                param.grad_sample = [g * scale.unsqueeze(0) for g in gs]
            else:
                param.grad_sample = gs * scale.unsqueeze(0)

    def save(self, path: str) -> None:
        """Save preconditioner state."""
        torch.save({
            'first_moment': self.first_moment,
            'second_moment': self.second_moment,
            'damping_per_layer': self.damping_per_layer,
            'base_damping': self.base_damping,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'step': self._step,
            'initialized': self._initialized,
        }, path)

    def load(self, path: str) -> None:
        """Load preconditioner state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.first_moment = checkpoint['first_moment']
        self.second_moment = checkpoint['second_moment']
        self.damping_per_layer = checkpoint.get('damping_per_layer', {})
        self.base_damping = checkpoint['base_damping']
        self.beta1 = checkpoint['beta1']
        self.beta2 = checkpoint['beta2']
        self._step = checkpoint['step']
        self._initialized = checkpoint['initialized']


class KFACPreconditioner:
    """
    K-FAC (Kronecker-Factored Approximate Curvature) preconditioner for DP-SGD.

    Unlike diagonal preconditioners (AdaDPS), K-FAC captures cross-parameter
    correlations by factoring the Fisher information matrix as a Kronecker
    product of activation covariance (A) and gradient covariance (G):

        F ≈ G ⊗ A

    Preconditioning then becomes:
        g_preconditioned = G^{-1/2} @ g @ A^{-1/2}

    This is applied per-sample BEFORE clipping in DP-SGD. The Kronecker factors
    are estimated from public/synthetic data (zero privacy cost).

    Supports nn.Linear and nn.Conv2d layers.
    """

    def __init__(
        self,
        damping: float = 1e-3,
        device: str = "cuda",
    ):
        self.damping = damping
        self.device = device

        # Kronecker factors (covariance matrices per layer)
        self.cov_A: Dict[str, torch.Tensor] = {}  # activation covariance
        self.cov_G: Dict[str, torch.Tensor] = {}  # gradient covariance

        # Inverse square roots (precomputed for fast apply)
        self.inv_A: Dict[str, torch.Tensor] = {}
        self.inv_G: Dict[str, torch.Tensor] = {}

        # Hook storage during estimation
        self._activations: Dict[str, torch.Tensor] = {}
        self._backprops: Dict[str, torch.Tensor] = {}
        self._handles = []

        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _register_hooks(self, model: nn.Module) -> None:
        """Register forward/backward hooks on Linear and Conv2d layers."""
        target = model._module if hasattr(model, '_module') else model

        for name, module in target.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            if not any(p.requires_grad for p in module.parameters()):
                continue

            def make_fwd_hook(layer_name):
                def hook(_mod, inp, _out):
                    self._activations[layer_name] = inp[0].detach()
                return hook

            def make_bwd_hook(layer_name):
                def hook(_mod, _grad_in, grad_out):
                    self._backprops[layer_name] = grad_out[0].detach()
                return hook

            self._handles.append(module.register_forward_hook(make_fwd_hook(name)))
            self._handles.append(module.register_full_backward_hook(make_bwd_hook(name)))

    def _remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._activations.clear()
        self._backprops.clear()

    def _compute_covariances_for_batch(self, model: nn.Module) -> None:
        """Compute and accumulate Kronecker factors from captured activations/backprops."""
        target = model._module if hasattr(model, '_module') else model
        name_to_module = dict(target.named_modules())

        for name in self._backprops:
            if name not in self._activations or name not in name_to_module:
                continue

            module = name_to_module[name]
            act = self._activations[name]
            grad = self._backprops[name]

            if isinstance(module, nn.Linear):
                A = act
                G = grad
                if A.dim() > 2:
                    A = A.reshape(-1, A.shape[-1])
                if G.dim() > 2:
                    G = G.reshape(-1, G.shape[-1])

                # Augment with bias column if module has bias
                if module.bias is not None:
                    A = torch.cat([A, torch.ones_like(A[:, :1])], dim=1)

                batch_A = (A.T @ A) / A.size(0)
                batch_G = (G.T @ G) / G.size(0)

            elif isinstance(module, nn.Conv2d):
                # im2col unfolding for activations
                X_unfold = F.unfold(
                    act, kernel_size=module.kernel_size,
                    padding=module.padding, stride=module.stride
                )
                X_unfold = X_unfold.transpose(1, 2).reshape(-1, X_unfold.size(1))

                if module.bias is not None:
                    X_unfold = torch.cat([X_unfold, torch.ones_like(X_unfold[:, :1])], dim=1)

                batch_A = (X_unfold.T @ X_unfold) / X_unfold.size(0)

                # Gradient output reshaped
                GY = grad.permute(0, 2, 3, 1).reshape(-1, grad.size(1))
                batch_G = (GY.T @ GY) / GY.size(0)
            else:
                continue

            # Accumulate
            if name in self.cov_A:
                self.cov_A[name] += batch_A
                self.cov_G[name] += batch_G
            else:
                self.cov_A[name] = batch_A
                self.cov_G[name] = batch_G

    def _compute_inverse_sqrts(self) -> None:
        """Compute (A + damping*I)^{-1/2} and (G + damping*I)^{-1/2} via eigendecomposition."""
        for name in self.cov_A:
            A = self.cov_A[name]
            A_damped = A + self.damping * torch.eye(A.size(0), device=A.device, dtype=A.dtype)
            eva, evc = torch.linalg.eigh(A_damped)
            self.inv_A[name] = evc @ torch.diag(eva.clamp(min=1e-6).rsqrt()) @ evc.T

            G = self.cov_G[name]
            G_damped = G + self.damping * torch.eye(G.size(0), device=G.device, dtype=G.dtype)
            evg, evcg = torch.linalg.eigh(G_damped)
            self.inv_G[name] = evcg @ torch.diag(evg.clamp(min=1e-6).rsqrt()) @ evcg.T

    def estimate_from_loader(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        num_steps: int,
        show_progress: bool = True,
        task: str = "segmentation",
    ) -> None:
        """Estimate K-FAC preconditioner from a data loader."""
        model.train()
        self.cov_A = {}
        self.cov_G = {}

        self._register_hooks(model)

        data_iter = iter(data_loader)
        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Estimating K-FAC preconditioner", leave=False)

        try:
            for _ in iterator:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    batch = next(data_iter)

                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    targets = batch.get("mask", batch.get("label")).to(self.device)
                else:
                    images, targets = batch
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                model.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                if loss.dim() > 0:
                    loss = loss.mean()
                loss.backward()

                self._compute_covariances_for_batch(model)
                self._activations.clear()
                self._backprops.clear()
                clear_grad_samples(model)
        finally:
            self._remove_hooks()

        # Average over batches
        for name in self.cov_A:
            self.cov_A[name] /= num_steps
            self.cov_G[name] /= num_steps

        # Compute inverse square roots
        self._compute_inverse_sqrts()

        model.zero_grad()
        clear_grad_samples(model)
        self._initialized = True

        n_layers = len(self.inv_A)
        total_size = sum(A.numel() + G.numel()
                        for A, G in zip(self.inv_A.values(), self.inv_G.values()))
        print(f"  K-FAC: {n_layers} layers, {total_size:,} factor elements")

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
        mask_type: str = "blob",
        mask_strategy: str = None,
        task: str = "classification",
        spatial_dims: int = 2,
        show_progress: bool = True,
    ) -> None:
        """Estimate K-FAC preconditioner from synthetic data."""
        if task == "classification":
            from ..data.synthetic import SyntheticClassificationDataset
            dataset = SyntheticClassificationDataset(
                num_classes=num_classes,
                n_samples=num_samples,
                image_size=image_size,
                in_channels=in_channels,
                noise_type=noise_type,
            )
        elif spatial_dims == 3:
            from ..data.synthetic3d import SyntheticSegmentation3DDataset
            if mask_strategy is None:
                mask_strategy_map = {"random": "random_shapes", "blob": "gaussian_blobs", "perlin": "pink_threshold"}
                mask_strategy = mask_strategy_map.get(mask_type, "gaussian_blobs")
            dataset = SyntheticSegmentation3DDataset(
                n_samples=num_samples, patch_size=image_size, in_channels=in_channels,
                num_classes=num_classes, noise_type=noise_type, mask_strategy=mask_strategy,
            )
        else:
            from ..data.synthetic import SyntheticSegmentationDataset
            if mask_strategy is None:
                mask_strategy_map = {"random": "random_shapes", "blob": "gaussian_blobs", "perlin": "pink_threshold"}
                mask_strategy = mask_strategy_map.get(mask_type, "gaussian_blobs")
            dataset = SyntheticSegmentationDataset(
                n_samples=num_samples, image_size=image_size, in_channels=in_channels,
                num_classes=num_classes, noise_type=noise_type, mask_strategy=mask_strategy,
            )

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        num_steps = min(len(data_loader), num_samples // batch_size)
        self.estimate_from_loader(
            model=model, data_loader=data_loader, loss_fn=loss_fn,
            num_steps=num_steps, show_progress=show_progress, task=task,
        )

    def apply(self, model: nn.Module) -> None:
        """
        Apply K-FAC preconditioning to per-sample gradients in-place.

        For each layer: g_preconditioned = inv_G @ g @ inv_A
        Applied via batched einsum on the per-sample grad_sample tensors.
        """
        if not self._initialized:
            raise RuntimeError("K-FAC preconditioner not initialized.")

        target = model._module if hasattr(model, '_module') else model

        for name, module in target.named_modules():
            if name not in self.inv_A or name not in self.inv_G:
                continue
            if not hasattr(module, 'weight') or not hasattr(module.weight, 'grad_sample'):
                continue
            if module.weight.grad_sample is None:
                continue

            g_sample_w = module.weight.grad_sample
            if isinstance(g_sample_w, list):
                g_sample_w = g_sample_w[-1]

            # For Conv2d: reshape [B, C_out, C_in, kH, kW] -> [B, C_out, C_in*kH*kW]
            is_conv = isinstance(module, nn.Conv2d)
            if is_conv:
                batch_size, out_ch = g_sample_w.shape[:2]
                g_sample_w = g_sample_w.view(batch_size, out_ch, -1)

            # Augment with bias gradient if present
            has_bias = (module.bias is not None and
                        hasattr(module.bias, 'grad_sample') and
                        module.bias.grad_sample is not None)
            if has_bias:
                g_sample_b = module.bias.grad_sample
                if isinstance(g_sample_b, list):
                    g_sample_b = g_sample_b[-1]
                g_sample_aug = torch.cat([g_sample_w, g_sample_b.unsqueeze(2)], dim=2)
            else:
                g_sample_aug = g_sample_w

            # K-FAC preconditioning: inv_G @ g @ inv_A
            temp = torch.einsum("oj,bjk->bok", self.inv_G[name], g_sample_aug)
            preconditioned = torch.einsum("bok,ki->boi", temp, self.inv_A[name])

            # Split bias back out
            if has_bias:
                new_w_grad = preconditioned[:, :, :-1]
                new_b_grad = preconditioned[:, :, -1]
            else:
                new_w_grad = preconditioned
                new_b_grad = None

            # Reshape Conv2d back
            if is_conv:
                new_w_grad = new_w_grad.view_as(module.weight.grad_sample)

            module.weight.grad_sample = new_w_grad
            if new_b_grad is not None:
                module.bias.grad_sample = new_b_grad

    def save(self, path: str) -> None:
        torch.save({
            'cov_A': self.cov_A,
            'cov_G': self.cov_G,
            'inv_A': self.inv_A,
            'inv_G': self.inv_G,
            'damping': self.damping,
            'initialized': self._initialized,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.cov_A = checkpoint['cov_A']
        self.cov_G = checkpoint['cov_G']
        self.inv_A = checkpoint['inv_A']
        self.inv_G = checkpoint['inv_G']
        self.damping = checkpoint['damping']
        self._initialized = checkpoint['initialized']


class ShampooPreconditioner:
    """
    Shampoo preconditioner for DP-SGD.

    Unlike K-FAC which estimates Fisher factors from activations and backprop
    signals (requiring forward/backward hooks), Shampoo computes preconditioners
    directly from gradient outer products:

        L = E[G @ G^T]   (left factor,  shape m x m)
        R = E[G^T @ G]   (right factor, shape n x n)

    Preconditioning:
        g_preconditioned = L^{-1/4} @ G @ R^{-1/4}

    The -1/4 power on each side gives an effective -1/2 power overall
    (equivalent to F^{-1/2} in the Kronecker approximation).

    Advantages over K-FAC for the public-data preconditioning paradigm:
    - No hooks needed: uses gradient tensors only
    - Covers ALL layer types: Conv2d, ConvTranspose2d, Linear, GroupNorm
    - More robust to distribution mismatch: gradient geometry transfers
      better than activation/backprop covariances across domains
    """

    def __init__(self, damping: float = 1e-4, device: str = "cuda"):
        self.damping = damping
        self.device = device

        # Gradient outer-product accumulators per parameter
        self.L: Dict[str, torch.Tensor] = {}   # G @ G^T
        self.R: Dict[str, torch.Tensor] = {}   # G^T @ G

        # Precomputed inverse 4th roots
        self.inv_L: Dict[str, torch.Tensor] = {}  # L^{-1/4}
        self.inv_R: Dict[str, torch.Tensor] = {}  # R^{-1/4}

        # Diagonal fallback for 1D params (biases, GroupNorm affine)
        self._diag: Dict[str, torch.Tensor] = {}

        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @staticmethod
    def _mat_pow_neg_quarter(M: torch.Tensor, damping: float) -> torch.Tensor:
        """Compute (M + damping * I)^{-1/4} via eigendecomposition."""
        M_d = M + damping * torch.eye(M.size(0), device=M.device, dtype=M.dtype)
        eigvals, eigvecs = torch.linalg.eigh(M_d)
        inv_q = eigvals.clamp(min=1e-7).pow(-0.25)
        return eigvecs @ torch.diag(inv_q) @ eigvecs.T

    def estimate_from_loader(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        num_steps: int,
        show_progress: bool = True,
        task: str = "segmentation",
    ) -> None:
        """Estimate Shampoo factors from a data loader (public or auxiliary data)."""
        model.train()
        target = model._module if hasattr(model, '_module') else model

        self.L.clear()
        self.R.clear()
        self._diag.clear()

        data_iter = iter(data_loader)
        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Estimating Shampoo preconditioner", leave=False)

        for _ in iterator:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                targets = batch.get("mask", batch.get("label")).to(self.device)
            else:
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)

            model.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()

            # Accumulate gradient outer products
            for name, param in target.named_parameters():
                if param.grad is None or not param.requires_grad:
                    continue

                g = param.grad.detach()

                if len(param.shape) >= 2:
                    # 2D+: reshape to (first_dim, remaining) for Shampoo
                    g_2d = g.reshape(g.shape[0], -1)
                    bL = g_2d @ g_2d.T          # (m, m)
                    bR = g_2d.T @ g_2d          # (n, n)

                    if name in self.L:
                        self.L[name] += bL
                        self.R[name] += bR
                    else:
                        self.L[name] = bL
                        self.R[name] = bR
                else:
                    # 1D: accumulate squared gradient for diagonal fallback
                    if name in self._diag:
                        self._diag[name] += g ** 2
                    else:
                        self._diag[name] = g.clone() ** 2

            clear_grad_samples(model)

        # Average over steps
        for name in self.L:
            self.L[name] /= num_steps
            self.R[name] /= num_steps
        for name in self._diag:
            self._diag[name] /= num_steps

        # Compute inverse 4th roots
        for name in self.L:
            self.inv_L[name] = self._mat_pow_neg_quarter(self.L[name], self.damping)
            self.inv_R[name] = self._mat_pow_neg_quarter(self.R[name], self.damping)

        # 1D params: simple inverse-scale  1/sqrt(E[g²] + damping)
        for name in self._diag:
            self._diag[name] = 1.0 / torch.sqrt(self._diag[name] + self.damping)

        model.zero_grad()
        clear_grad_samples(model)
        self._initialized = True

        n_shampoo = len(self.inv_L)
        n_diag = len(self._diag)
        total_size = sum(L.numel() + R.numel()
                         for L, R in zip(self.inv_L.values(), self.inv_R.values()))
        print(f"  Shampoo: {n_shampoo} layers (matrix), {n_diag} layers (diagonal), "
              f"{total_size:,} factor elements")

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
        mask_type: str = "blob",
        mask_strategy: str = None,
        task: str = "classification",
        spatial_dims: int = 2,
        show_progress: bool = True,
    ) -> None:
        """Estimate Shampoo preconditioner from synthetic data."""
        if task == "classification":
            from ..data.synthetic import SyntheticClassificationDataset
            dataset = SyntheticClassificationDataset(
                num_classes=num_classes,
                n_samples=num_samples,
                image_size=image_size,
                in_channels=in_channels,
                noise_type=noise_type,
            )
        elif spatial_dims == 3:
            from ..data.synthetic3d import SyntheticSegmentation3DDataset
            if mask_strategy is None:
                mask_strategy_map = {"random": "random_shapes", "blob": "gaussian_blobs", "perlin": "pink_threshold"}
                mask_strategy = mask_strategy_map.get(mask_type, "gaussian_blobs")
            dataset = SyntheticSegmentation3DDataset(
                n_samples=num_samples, patch_size=image_size, in_channels=in_channels,
                num_classes=num_classes, noise_type=noise_type, mask_strategy=mask_strategy,
            )
        else:
            from ..data.synthetic import SyntheticSegmentationDataset
            if mask_strategy is None:
                mask_strategy_map = {"random": "random_shapes", "blob": "gaussian_blobs", "perlin": "pink_threshold"}
                mask_strategy = mask_strategy_map.get(mask_type, "gaussian_blobs")
            dataset = SyntheticSegmentationDataset(
                n_samples=num_samples, image_size=image_size, in_channels=in_channels,
                num_classes=num_classes, noise_type=noise_type, mask_strategy=mask_strategy,
            )

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        num_steps = min(len(data_loader), num_samples // batch_size)
        self.estimate_from_loader(
            model=model, data_loader=data_loader, loss_fn=loss_fn,
            num_steps=num_steps, show_progress=show_progress, task=task,
        )

    def apply(self, model: nn.Module) -> None:
        """
        Apply Shampoo preconditioning to per-sample gradients in-place.

        For 2D+ params:  g_precond = inv_L @ g_2d @ inv_R   (reshaped back)
        For 1D params:   g_precond = g * diag_scale
        """
        if not self._initialized:
            raise RuntimeError("Shampoo preconditioner not initialized.")

        target = model._module if hasattr(model, '_module') else model

        for name, param in target.named_parameters():
            gs = getattr(param, 'grad_sample', None)
            if gs is None:
                continue
            if isinstance(gs, list):
                gs = gs[-1]

            if name in self.inv_L:
                # 2D+: full Shampoo  inv_L @ G @ inv_R
                original_shape = gs.shape            # (B, *param_shape)
                batch_size = original_shape[0]
                first_dim = original_shape[1]

                # Reshape to (B, m, n)
                g_2d = gs.reshape(batch_size, first_dim, -1)

                # Batched matrix multiply via einsum
                temp = torch.einsum("ij,bjk->bik", self.inv_L[name], g_2d)
                preconditioned = torch.einsum("bij,jk->bik", temp, self.inv_R[name])

                param.grad_sample = preconditioned.reshape(original_shape)

            elif name in self._diag:
                # 1D: element-wise scaling
                param.grad_sample = gs * self._diag[name]

    def save(self, path: str) -> None:
        torch.save({
            'L': self.L, 'R': self.R,
            'inv_L': self.inv_L, 'inv_R': self.inv_R,
            'diag': self._diag,
            'damping': self.damping,
            'initialized': self._initialized,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.L = checkpoint['L']
        self.R = checkpoint['R']
        self.inv_L = checkpoint['inv_L']
        self.inv_R = checkpoint['inv_R']
        self._diag = checkpoint['diag']
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
    damping: float = 0.1,
    beta1: float = 0.0,
    beta2: float = 0.9,
    adaptive_damping: bool = True,
    device: str = "cuda",
):
    """
    Factory function to create preconditioner.

    Args:
        precond_type: "none", "adadps", "momentum", "adadps_public", etc.
        damping: Damping factor
        beta1: First moment decay (for momentum preconditioner)
        beta2: Second moment decay (for momentum preconditioner)
        adaptive_damping: Scale damping per-layer (for momentum)
        device: Device for tensors

    Returns:
        Preconditioner instance
    """
    if precond_type == "none":
        return IdentityPreconditioner()
    elif precond_type in ["adadps", "adadps_public", "adadps_private", "adadps_synthetic"]:
        return AdaDPSPreconditioner(damping=damping, device=device)
    elif precond_type in ["momentum", "momentum_synthetic", "momentum_public"]:
        return MomentumPreconditioner(
            damping=damping,
            beta1=beta1,
            beta2=beta2,
            adaptive_damping=adaptive_damping,
            device=device,
        )
    elif precond_type in ["kfac", "kfac_public", "kfac_synthetic"]:
        kfac_damping = damping if damping != 0.1 else 1e-3
        return KFACPreconditioner(damping=kfac_damping, device=device)
    elif precond_type in ["shampoo", "shampoo_public", "shampoo_synthetic"]:
        return ShampooPreconditioner(damping=1e-4, device=device)
    else:
        raise ValueError(f"Unknown preconditioner type: {precond_type}")

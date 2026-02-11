"""
Synthetic Data Generation for Preconditioner Estimation.

This module provides synthetic image-mask generation for estimating
AdaDPS preconditioners without requiring real public data.

Includes:
- Noise generators (white, pink, brown, Perlin)
- Synthetic mask generators (blobs, shapes, patterns)
- SyntheticDataset class for PyTorch integration
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, Literal, List
from scipy import ndimage
from scipy.spatial import Voronoi
from skimage.draw import disk, polygon

# Try to import noise for Perlin, fallback to simple implementation
try:
    from noise import pnoise2
    HAS_PERLIN = True
except ImportError:
    HAS_PERLIN = False


# =============================================================================
# NOISE GENERATORS
# =============================================================================

def generate_white_noise(size: int) -> np.ndarray:
    """
    Generate white noise (flat spectrum).

    Args:
        size: Image size (square)

    Returns:
        Normalized noise array of shape (size, size)
    """
    noise = np.random.randn(size, size)
    return (noise - noise.mean()) / (noise.std() + 1e-8)


def generate_pink_noise(size: int) -> np.ndarray:
    """
    Generate pink noise (1/f spectrum).

    Pink noise has natural image-like statistics and is a good
    default for most synthetic data generation.

    Args:
        size: Image size (square)

    Returns:
        Normalized noise array of shape (size, size)
    """
    white = np.fft.fft2(np.random.randn(size, size))

    # Create 1/f filter
    freq_x = np.fft.fftfreq(size)
    freq_y = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freq_x, freq_y)
    freq = np.sqrt(fx**2 + fy**2)
    freq[0, 0] = 1  # Avoid division by zero

    # Apply 1/f filter
    pink = white / freq
    pink[0, 0] = 0  # Zero DC component

    result = np.real(np.fft.ifft2(pink))
    return (result - result.mean()) / (result.std() + 1e-8)


def generate_brown_noise(size: int) -> np.ndarray:
    """
    Generate brown noise (1/f² spectrum).

    Brown noise is smoother than pink noise, good for soft tissue.

    Args:
        size: Image size (square)

    Returns:
        Normalized noise array of shape (size, size)
    """
    white = np.fft.fft2(np.random.randn(size, size))

    freq_x = np.fft.fftfreq(size)
    freq_y = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freq_x, freq_y)
    freq = np.sqrt(fx**2 + fy**2)
    freq[0, 0] = 1

    brown = white / (freq ** 2)
    brown[0, 0] = 0

    result = np.real(np.fft.ifft2(brown))
    return (result - result.mean()) / (result.std() + 1e-8)


def generate_perlin_noise(size: int, scale: float = 50, octaves: int = 4) -> np.ndarray:
    """
    Generate Perlin noise (coherent, organic patterns).

    Args:
        size: Image size (square)
        scale: Noise scale (larger = smoother)
        octaves: Number of octaves (more = more detail)

    Returns:
        Normalized noise array of shape (size, size)
    """
    if not HAS_PERLIN:
        # Fallback to filtered noise if perlin library not available
        return generate_pink_noise(size)

    noise = np.zeros((size, size))
    offset_x, offset_y = np.random.rand(2) * 1000

    for i in range(size):
        for j in range(size):
            noise[i, j] = pnoise2(
                (i + offset_x) / scale,
                (j + offset_y) / scale,
                octaves=octaves
            )

    return (noise - noise.mean()) / (noise.std() + 1e-8)


def get_noise_generator(noise_type: str) -> Callable:
    """
    Get noise generator function by name.

    Args:
        noise_type: One of "white", "pink", "brown", "perlin"

    Returns:
        Noise generator function
    """
    generators = {
        "white": generate_white_noise,
        "pink": generate_pink_noise,
        "brown": generate_brown_noise,
        "perlin": generate_perlin_noise,
    }
    if noise_type not in generators:
        raise ValueError(f"Unknown noise type: {noise_type}. Available: {list(generators.keys())}")
    return generators[noise_type]


# =============================================================================
# MASK GENERATORS - Classification (for MedMNIST-like tasks)
# =============================================================================

def generate_random_label(num_classes: int) -> int:
    """Generate a random class label."""
    return np.random.randint(0, num_classes)


def generate_class_from_noise_stats(noise: np.ndarray, num_classes: int) -> int:
    """
    Generate a class label based on noise statistics.

    Uses properties of the generated noise to determine class,
    providing some structure to the synthetic supervision.
    """
    # Use mean and std to determine class
    features = np.array([noise.mean(), noise.std(), noise.max(), noise.min()])
    hash_val = int(np.abs(np.sum(features * 1000))) % num_classes
    return hash_val


# =============================================================================
# MASK GENERATORS - Segmentation
# =============================================================================

def generate_gaussian_blobs(
    size: int,
    n_blobs: int = 5,
    size_range: Tuple[int, int] = (5, 20),
) -> np.ndarray:
    """
    Generate soft elliptical blobs.

    Good for: lesions, nodules, round structures.

    Args:
        size: Image size
        n_blobs: Number of blobs
        size_range: (min, max) blob radius

    Returns:
        Binary mask of shape (size, size)
    """
    mask = np.zeros((size, size))

    for _ in range(n_blobs):
        # Random center (avoid edges)
        margin = size_range[1] + 5
        cx = np.random.randint(margin, size - margin)
        cy = np.random.randint(margin, size - margin)

        # Random ellipse parameters
        rx = np.random.randint(*size_range)
        ry = np.random.randint(*size_range)
        angle = np.random.rand() * 2 * np.pi

        # Create ellipse mask
        y, x = np.ogrid[:size, :size]
        x_rot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
        y_rot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)

        ellipse = (x_rot / (rx + 1e-6)) ** 2 + (y_rot / (ry + 1e-6)) ** 2 <= 1
        mask = np.maximum(mask, ellipse.astype(float))

    return mask


def generate_random_shapes(
    size: int,
    n_shapes: int = 5,
    size_range: Tuple[int, int] = (3, 15),
) -> np.ndarray:
    """
    Generate random geometric shapes (circles, rectangles).

    Good for: general synthetic supervision.

    Args:
        size: Image size
        n_shapes: Number of shapes
        size_range: (min, max) shape size

    Returns:
        Binary mask of shape (size, size)
    """
    mask = np.zeros((size, size))

    for _ in range(n_shapes):
        shape_type = np.random.choice(["circle", "rectangle"])
        margin = size_range[1] + 5
        cx = np.random.randint(margin, size - margin)
        cy = np.random.randint(margin, size - margin)
        r = np.random.randint(*size_range)

        if shape_type == "circle":
            rr, cc = disk((cy, cx), r, shape=(size, size))
            mask[rr, cc] = 1
        else:
            # Rectangle
            w = np.random.randint(*size_range)
            h = np.random.randint(*size_range)
            x1, x2 = max(0, cx - w), min(size, cx + w)
            y1, y2 = max(0, cy - h), min(size, cy + h)
            mask[y1:y2, x1:x2] = 1

    return mask


def generate_pink_threshold(
    size: int,
    threshold_percentile: float = 50,
) -> np.ndarray:
    """
    Generate mask by thresholding pink noise.

    Simplest approach - creates random blob-like regions.

    Args:
        size: Image size
        threshold_percentile: Percentile for thresholding

    Returns:
        Binary mask of shape (size, size)
    """
    noise = generate_pink_noise(size)
    threshold = np.percentile(noise, threshold_percentile)
    return (noise > threshold).astype(float)


def generate_voronoi_cells(
    size: int,
    n_cells: int = 20,
    shrink: float = 0.8,
) -> np.ndarray:
    """
    Generate cell-like regions using Voronoi tessellation.

    Good for: cell segmentation, histopathology.

    Args:
        size: Image size
        n_cells: Number of cells
        shrink: Shrinkage factor (1.0 = no gap, 0.5 = large gaps)

    Returns:
        Binary mask of shape (size, size)
    """
    # Random cell centers
    points = np.random.rand(n_cells, 2) * size

    # Add boundary points to close Voronoi regions
    boundary = np.array([
        [-size, -size], [-size, size*2], [size*2, -size], [size*2, size*2]
    ])
    all_points = np.vstack([points, boundary])

    try:
        vor = Voronoi(all_points)
    except Exception:
        # Fallback to simple circles if Voronoi fails
        return generate_gaussian_blobs(size, n_blobs=n_cells)

    mask = np.zeros((size, size))

    for region_idx in vor.point_region[:n_cells]:
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue

        vertices = vor.vertices[region]
        centroid = vertices.mean(axis=0)
        vertices = centroid + shrink * (vertices - centroid)
        vertices = np.clip(vertices, 0, size - 1)

        try:
            rr, cc = polygon(vertices[:, 1], vertices[:, 0], shape=(size, size))
            mask[rr, cc] = 1
        except Exception:
            continue

    return mask


def generate_frangi_vessels(
    size: int,
    max_depth: int = 6,
    branch_prob: float = 0.4,
    min_branch_gap: int = 4,
    radius_decay: float = 0.79,
    momentum: float = 0.85,
) -> np.ndarray:
    """
    Generate interconnected vascular tree structures via skeleton + distance transform.

    Grows branching trees with direction-based momentum (smooth curves),
    Murray's Law radius decay at bifurcations, and rasterizes using
    distance transform for smooth, non-lumpy tubes.

    Args:
        size: Image size
        max_depth: Maximum branching depth
        branch_prob: Probability of branching at each eligible step
        min_branch_gap: Minimum steps between consecutive branches
        radius_decay: Radius multiplier at branch (0.79 ≈ Murray's Law)
        momentum: Direction smoothing (0=random walk, 1=straight line)

    Returns:
        Binary mask of shape (size, size)
    """
    from scipy.ndimage import distance_transform_edt

    segments = []  # List of (start, end, radius) tuples
    occupied = np.zeros((size, size), dtype=bool)  # Collision map
    init_radius = max(1.5, size / 60)
    step_len = max(2.0, size / 50)

    def _mark_occupied(p_start, p_end, radius):
        """Rasterize a tube segment into the occupied grid (half-radius zone)."""
        seg_len = np.linalg.norm(p_end - p_start)
        n_pts = max(2, int(seg_len * 2))
        mark_r = max(1.0, radius * 0.5)
        ri = int(np.ceil(mark_r))
        for t in np.linspace(0, 1, n_pts):
            pt = p_start + t * (p_end - p_start)
            ix = int(np.clip(pt[0], 0, size - 1))
            iy = int(np.clip(pt[1], 0, size - 1))
            y0, y1 = max(0, iy - ri), min(size, iy + ri + 1)
            x0, x1 = max(0, ix - ri), min(size, ix + ri + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            occupied[y0:y1, x0:x1][(yy - iy)**2 + (xx - ix)**2 <= mark_r**2] = True

    def _is_occupied(pos):
        """Check if a position's centerline crosses an existing vessel."""
        ix = int(np.clip(pos[0], 0, size - 1))
        iy = int(np.clip(pos[1], 0, size - 1))
        return occupied[iy, ix]

    def _grow(pos, direction, radius, depth, max_steps):
        if depth > max_depth or radius < 0.5:
            return
        n_steps = np.random.randint(max(4, max_steps // 2), max_steps + 1)
        steps_since_branch = min_branch_gap  # allow branching from the start
        for s in range(n_steps):
            # Momentum-based direction update: smooth curves, not Brownian jitter
            perturbation = np.random.randn(2) * 0.4
            direction = momentum * direction + (1 - momentum) * perturbation
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            new_pos = pos + direction * step_len

            # Reflect off boundaries
            for dim in range(2):
                if new_pos[dim] < 1 or new_pos[dim] >= size - 1:
                    direction[dim] = -direction[dim]
                    new_pos[dim] = np.clip(new_pos[dim], 1, size - 2)

            # Collision check: stop if centerline enters existing vessel interior
            # Grace period of 3 steps so branches can escape parent's zone
            if s > 2 and _is_occupied(new_pos):
                break

            segments.append((pos.copy(), new_pos.copy(), radius))
            _mark_occupied(pos, new_pos, radius)
            pos = new_pos.copy()
            steps_since_branch += 1

            # Branch (Murray's Law) — enforce minimum gap between branches
            if s > 2 and steps_since_branch >= min_branch_gap and np.random.random() < branch_prob:
                branch_angle = np.random.choice([-1, 1]) * np.random.uniform(0.4, 1.0)
                cos_a, sin_a = np.cos(branch_angle), np.sin(branch_angle)
                branch_dir = np.array([
                    direction[0] * cos_a - direction[1] * sin_a,
                    direction[0] * sin_a + direction[1] * cos_a,
                ])
                _grow(pos, branch_dir, radius * radius_decay, depth + 1,
                      max(4, max_steps - 3))
                steps_since_branch = 0

    # Single root, randomly placed — initial direction toward image center
    pos = np.array([
        np.random.uniform(size * 0.2, size * 0.8),
        np.random.uniform(size * 0.2, size * 0.8),
    ])
    center = np.array([size / 2.0, size / 2.0])
    d = center - pos
    d = d / (np.linalg.norm(d) + 1e-8)
    # Add slight random perturbation so it's not always dead-center
    d = d + np.random.randn(2) * 0.2
    d = d / (np.linalg.norm(d) + 1e-8)
    _grow(pos, d, init_radius, 0, max_steps=25)

    # Rasterize: draw skeleton lines with per-segment radius stored in a radius map
    skeleton = np.zeros((size, size), dtype=bool)
    radius_map = np.zeros((size, size), dtype=np.float64)

    for p_start, p_end, radius in segments:
        seg_len = np.linalg.norm(p_end - p_start)
        n_pts = max(2, int(seg_len * 2))
        for t in np.linspace(0, 1, n_pts):
            pt = p_start + t * (p_end - p_start)
            ix, iy = int(np.clip(pt[0], 0, size - 1)), int(np.clip(pt[1], 0, size - 1))
            skeleton[iy, ix] = True
            radius_map[iy, ix] = max(radius_map[iy, ix], radius)

    if not skeleton.any():
        return np.zeros((size, size), dtype=np.float64)

    # Distance transform: for each pixel, get distance to nearest skeleton point
    # AND the indices of that nearest skeleton point (to look up its radius)
    dist, nearest_idx = distance_transform_edt(~skeleton, return_distances=True,
                                                return_indices=True)

    # Look up the radius at the nearest skeleton point for each pixel
    nearest_radius = radius_map[nearest_idx[0], nearest_idx[1]]

    # Mask: pixel is inside vessel if its distance to nearest skeleton <= that point's radius
    mask = (dist <= nearest_radius).astype(np.float64)

    return mask


def get_mask_generator(strategy: str) -> Callable:
    """
    Get mask generator function by name.

    Args:
        strategy: One of "gaussian_blobs", "random_shapes", "pink_threshold",
                  "voronoi", "frangi"

    Returns:
        Mask generator function
    """
    generators = {
        "gaussian_blobs": generate_gaussian_blobs,
        "random_shapes": generate_random_shapes,
        "pink_threshold": generate_pink_threshold,
        "voronoi": generate_voronoi_cells,
        "frangi": generate_frangi_vessels,
    }
    if strategy not in generators:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(generators.keys())}")
    return generators[strategy]


# =============================================================================
# SYNTHETIC DATASET CLASS
# =============================================================================

class SyntheticClassificationDataset(Dataset):
    """
    Synthetic dataset for classification tasks (like MedMNIST).

    Generates noise images with random labels for preconditioner estimation.
    """

    def __init__(
        self,
        num_classes: int = 9,
        n_samples: int = 1000,
        image_size: int = 28,
        in_channels: int = 3,
        noise_type: str = "pink",
        label_strategy: str = "random",
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            num_classes: Number of classes
            n_samples: Number of synthetic samples to generate
            image_size: Image size (square)
            in_channels: Number of channels (1 or 3)
            noise_type: Type of noise ("white", "pink", "brown", "perlin")
            label_strategy: "random" or "noise_stats"
            transform: Optional transform
            seed: Random seed for reproducibility
        """
        self.num_classes = num_classes
        self.n_samples = n_samples
        self.image_size = image_size
        self.in_channels = in_channels
        self.noise_type = noise_type
        self.label_strategy = label_strategy
        self.transform = transform

        if seed is not None:
            np.random.seed(seed)

        self.noise_fn = get_noise_generator(noise_type)

        # Pre-generate samples for consistency
        self.images = []
        self.labels = []

        for _ in range(n_samples):
            # Generate noise image
            noise = self.noise_fn(image_size)

            # Normalize to [0, 1]
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

            # Convert to multi-channel if needed
            if in_channels == 3:
                noise = np.stack([noise, noise, noise], axis=-1)
            else:
                noise = noise[:, :, np.newaxis]

            # Generate label
            if label_strategy == "random":
                label = generate_random_label(num_classes)
            else:
                label = generate_class_from_noise_stats(noise, num_classes)

            self.images.append(noise.astype(np.float32))
            self.labels.append(label)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            try:
                transformed = self.transform(image=image)
                image = transformed["image"]
            except (TypeError, KeyError):
                image = self.transform(image)

        # Convert to tensor [C, H, W]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        label = torch.tensor(label, dtype=torch.long)

        return image, label


def composite_image_with_mask(
    background: np.ndarray,
    mask: np.ndarray,
    fg_intensity_shift: float = 0.3,
    fg_noise_scale: float = 0.15,
    edge_width: int = 2,
    edge_brightness: float = 0.2,
) -> np.ndarray:
    """
    Make masked regions visually distinct in the synthetic image.

    Applies:
    1. Intensity shift inside masked region (brighter/darker foreground)
    2. Different noise texture inside mask
    3. Edge highlighting at mask boundaries

    Works for both 2D and 3D arrays. For multi-channel images, apply per-channel.

    Args:
        background: noise image, normalized [0, 1]
        mask: binary mask (same spatial dims as background)
        fg_intensity_shift: intensity offset for foreground (can be negative)
        fg_noise_scale: scale of additional foreground noise
        edge_width: dilation iterations for edge detection
        edge_brightness: intensity boost at edges

    Returns:
        Composited image, same shape as background, clipped to [0, 1]
    """
    result = background.copy()
    mask_bool = mask > 0.5

    # 1. Intensity shift in foreground
    result[mask_bool] += fg_intensity_shift

    # 2. Add different-scale noise inside foreground
    fg_noise = np.random.randn(*background.shape) * fg_noise_scale
    result[mask_bool] += fg_noise[mask_bool]

    # 3. Edge highlighting via morphological gradient
    dilated = ndimage.binary_dilation(mask_bool, iterations=edge_width)
    eroded = ndimage.binary_erosion(mask_bool, iterations=max(1, edge_width // 2))
    edge = dilated & ~eroded
    result[edge] += edge_brightness

    return np.clip(result, 0, 1)


class SyntheticSegmentationDataset(Dataset):
    """
    Synthetic dataset for segmentation tasks.

    Generates noise images with synthetic masks for preconditioner estimation.
    Foreground regions are visually distinct from background.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        image_size: int = 256,
        in_channels: int = 3,
        num_classes: int = 1,
        noise_type: str = "pink",
        mask_strategy: str = "gaussian_blobs",
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        realistic: bool = True,
        **mask_kwargs,
    ):
        """
        Args:
            n_samples: Number of synthetic samples
            image_size: Image size (square)
            in_channels: Number of channels (1 or 3)
            num_classes: Number of output mask channels (1 for binary, >1 for multi-label)
            noise_type: Type of noise
            mask_strategy: Mask generation strategy
            transform: Optional transform (albumentations-style)
            seed: Random seed
            realistic: If True, make masked regions visually distinct
            **mask_kwargs: Additional arguments for mask generator
        """
        self.n_samples = n_samples
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.noise_type = noise_type
        self.mask_strategy = mask_strategy
        self.transform = transform
        self.mask_kwargs = mask_kwargs
        self.realistic = realistic

        if seed is not None:
            np.random.seed(seed)

        self.noise_fn = get_noise_generator(noise_type)
        self.mask_fn = get_mask_generator(mask_strategy)

        # Pre-generate samples
        self.images = []
        self.masks = []

        for _ in range(n_samples):
            # Generate noise image
            noise = self.noise_fn(image_size)
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

            # Generate mask(s)
            if num_classes > 1:
                mask_channels = []
                for _ in range(num_classes):
                    m = self.mask_fn(image_size, **mask_kwargs)
                    mask_channels.append(m)
                mask = np.stack(mask_channels, axis=0)  # (C, H, W)
                mask_composite = (mask.max(axis=0) > 0.5).astype(np.float32)
            else:
                mask = self.mask_fn(image_size, **mask_kwargs)
                mask_composite = mask

            # Make foreground visually distinct
            if realistic:
                shift = np.random.uniform(0.15, 0.4) * np.random.choice([-1, 1])
                noise = composite_image_with_mask(
                    noise, mask_composite,
                    fg_intensity_shift=shift,
                    fg_noise_scale=0.1,
                    edge_width=max(1, image_size // 64),
                    edge_brightness=0.15,
                )

            if in_channels == 3:
                noise = np.stack([noise, noise, noise], axis=-1)
            else:
                noise = noise[:, :, np.newaxis]

            self.images.append(noise.astype(np.float32))
            self.masks.append(mask.astype(np.float32))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform is not None:
            try:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            except (TypeError, KeyError):
                pass

        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)

        return image, mask


def create_synthetic_dataset(
    task: Literal["classification", "segmentation"],
    **kwargs,
) -> Dataset:
    """
    Factory function to create synthetic dataset.

    Args:
        task: "classification" or "segmentation"
        **kwargs: Arguments for the dataset class

    Returns:
        SyntheticClassificationDataset or SyntheticSegmentationDataset
    """
    if task == "classification":
        return SyntheticClassificationDataset(**kwargs)
    elif task == "segmentation":
        return SyntheticSegmentationDataset(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


# Quick test
if __name__ == "__main__":
    print("Testing noise generators...")
    for noise_type in ["white", "pink", "brown"]:
        noise = get_noise_generator(noise_type)(64)
        print(f"  {noise_type}: shape={noise.shape}, mean={noise.mean():.3f}, std={noise.std():.3f}")

    print("\nTesting mask generators...")
    for strategy in ["gaussian_blobs", "random_shapes", "pink_threshold"]:
        mask = get_mask_generator(strategy)(64)
        coverage = mask.mean() * 100
        print(f"  {strategy}: shape={mask.shape}, coverage={coverage:.1f}%")

    print("\nTesting SyntheticClassificationDataset...")
    dataset = SyntheticClassificationDataset(
        num_classes=9, n_samples=100, image_size=28, noise_type="pink"
    )
    image, label = dataset[0]
    print(f"  Image: {image.shape}, Label: {label}")

    print("\nTesting SyntheticSegmentationDataset...")
    dataset = SyntheticSegmentationDataset(
        n_samples=100, image_size=64, mask_strategy="gaussian_blobs"
    )
    image, mask = dataset[0]
    print(f"  Image: {image.shape}, Mask: {mask.shape}")

    print("\nSynthetic data generation working!")

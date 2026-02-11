"""
3D Synthetic Data Generation for Preconditioner Estimation.

Mirrors synthetic.py for 3D volumes. On-the-fly generation (too large
to pre-generate). Deterministic per-index seeding.

Includes all generators from 2D synthetic.py ported to 3D:
- Noise: white, pink, brown, perlin
- Masks: gaussian_blobs (ellipsoids), random_shapes, pink_threshold,
         voronoi, frangi
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Tuple
from scipy import ndimage

# Try to import noise for Perlin, fallback to filtered noise
try:
    from noise import pnoise3
    HAS_PERLIN_3D = True
except ImportError:
    HAS_PERLIN_3D = False


# =============================================================================
# NOISE GENERATORS (3D)
# =============================================================================

def generate_white_noise_3d(size: int) -> np.ndarray:
    """Generate 3D white noise (flat spectrum)."""
    noise = np.random.randn(size, size, size)
    return (noise - noise.mean()) / (noise.std() + 1e-8)


def generate_pink_noise_3d(size: int) -> np.ndarray:
    """Generate 3D pink noise (1/f spectrum)."""
    white = np.fft.fftn(np.random.randn(size, size, size))

    freq_x = np.fft.fftfreq(size)
    freq_y = np.fft.fftfreq(size)
    freq_z = np.fft.fftfreq(size)
    fx, fy, fz = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
    freq = np.sqrt(fx**2 + fy**2 + fz**2)
    freq[0, 0, 0] = 1  # Avoid division by zero

    pink = white / freq
    pink[0, 0, 0] = 0  # Zero DC component

    result = np.real(np.fft.ifftn(pink))
    return (result - result.mean()) / (result.std() + 1e-8)


def generate_brown_noise_3d(size: int) -> np.ndarray:
    """Generate 3D brown noise (1/f^2 spectrum)."""
    white = np.fft.fftn(np.random.randn(size, size, size))

    freq_x = np.fft.fftfreq(size)
    freq_y = np.fft.fftfreq(size)
    freq_z = np.fft.fftfreq(size)
    fx, fy, fz = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
    freq = np.sqrt(fx**2 + fy**2 + fz**2)
    freq[0, 0, 0] = 1

    brown = white / (freq ** 2)
    brown[0, 0, 0] = 0

    result = np.real(np.fft.ifftn(brown))
    return (result - result.mean()) / (result.std() + 1e-8)


def generate_perlin_noise_3d(size: int, scale: float = 30, octaves: int = 4) -> np.ndarray:
    """
    Generate 3D Perlin noise (coherent, organic patterns).

    Falls back to pink noise if pnoise3 is not available.
    """
    if not HAS_PERLIN_3D:
        return generate_pink_noise_3d(size)

    noise = np.zeros((size, size, size))
    offset_x, offset_y, offset_z = np.random.rand(3) * 1000

    for i in range(size):
        for j in range(size):
            for k in range(size):
                noise[i, j, k] = pnoise3(
                    (i + offset_x) / scale,
                    (j + offset_y) / scale,
                    (k + offset_z) / scale,
                    octaves=octaves,
                )

    return (noise - noise.mean()) / (noise.std() + 1e-8)


def get_noise_generator_3d(noise_type: str) -> Callable:
    """Get 3D noise generator function by name."""
    generators = {
        "white": generate_white_noise_3d,
        "pink": generate_pink_noise_3d,
        "brown": generate_brown_noise_3d,
        "perlin": generate_perlin_noise_3d,
    }
    if noise_type not in generators:
        raise ValueError(f"Unknown noise type: {noise_type}. Available: {list(generators.keys())}")
    return generators[noise_type]


# =============================================================================
# MASK GENERATORS (3D)
# =============================================================================

def generate_gaussian_blobs_3d(
    size: int,
    n_blobs: int = 5,
    size_range: Tuple[int, int] = (3, 12),
) -> np.ndarray:
    """
    Generate random 3D ellipsoids as segmentation mask.

    Good for: tumor-like structures in brain MRI.
    """
    mask = np.zeros((size, size, size))
    margin = size_range[1] + 3

    for _ in range(n_blobs):
        cx = np.random.randint(margin, max(margin + 1, size - margin))
        cy = np.random.randint(margin, max(margin + 1, size - margin))
        cz = np.random.randint(margin, max(margin + 1, size - margin))

        rx = np.random.randint(*size_range)
        ry = np.random.randint(*size_range)
        rz = np.random.randint(*size_range)

        z, y, x = np.ogrid[:size, :size, :size]
        ellipsoid = (
            ((x - cx) / (rx + 1e-6)) ** 2
            + ((y - cy) / (ry + 1e-6)) ** 2
            + ((z - cz) / (rz + 1e-6)) ** 2
        ) <= 1
        mask = np.maximum(mask, ellipsoid.astype(float))

    return mask


def generate_random_shapes_3d(
    size: int,
    n_shapes: int = 5,
    size_range: Tuple[int, int] = (3, 12),
) -> np.ndarray:
    """
    Generate random 3D geometric shapes (spheres + boxes).

    Good for: general synthetic supervision.
    """
    mask = np.zeros((size, size, size))

    for _ in range(n_shapes):
        shape_type = np.random.choice(["sphere", "box"])
        margin = size_range[1] + 3
        cx = np.random.randint(margin, max(margin + 1, size - margin))
        cy = np.random.randint(margin, max(margin + 1, size - margin))
        cz = np.random.randint(margin, max(margin + 1, size - margin))

        if shape_type == "sphere":
            r = np.random.randint(*size_range)
            z, y, x = np.ogrid[:size, :size, :size]
            sphere = ((x - cx)**2 + (y - cy)**2 + (z - cz)**2) <= r**2
            mask = np.maximum(mask, sphere.astype(float))
        else:
            wx = np.random.randint(*size_range)
            wy = np.random.randint(*size_range)
            wz = np.random.randint(*size_range)
            x1, x2 = max(0, cx - wx), min(size, cx + wx)
            y1, y2 = max(0, cy - wy), min(size, cy + wy)
            z1, z2 = max(0, cz - wz), min(size, cz + wz)
            mask[z1:z2, y1:y2, x1:x2] = 1

    return mask


def generate_pink_threshold_3d(
    size: int,
    threshold_percentile: float = 50,
) -> np.ndarray:
    """
    Generate mask by thresholding 3D pink noise.

    Creates random blob-like 3D regions.
    """
    noise = generate_pink_noise_3d(size)
    threshold = np.percentile(noise, threshold_percentile)
    return (noise > threshold).astype(float)


def generate_voronoi_cells_3d(
    size: int,
    n_cells: int = 15,
    shrink: float = 0.8,
) -> np.ndarray:
    """
    Generate cell-like 3D regions using nearest-neighbor Voronoi.

    Uses distance-based assignment with erosion for cell gaps.
    """
    # Random cell centers
    centers = np.random.rand(n_cells, 3) * size

    # Create coordinate grid
    coords = np.stack(np.mgrid[:size, :size, :size], axis=-1).reshape(-1, 3)

    # Assign each voxel to nearest center
    dists = np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=-1)
    labels = np.argmin(dists, axis=1).reshape(size, size, size)

    # Randomly select ~50% of cells as foreground
    fg_cells = set(np.random.choice(n_cells, n_cells // 2, replace=False))
    mask = np.isin(labels, list(fg_cells)).astype(float)

    # Erode to create gaps between cells
    if shrink < 1.0:
        erode_iters = max(1, int((1 - shrink) * 3))
        mask = ndimage.binary_erosion(mask, iterations=erode_iters).astype(float)

    return mask


def generate_frangi_vessels_3d(
    size: int,
    max_depth: int = 6,
    branch_prob: float = 0.4,
    min_branch_gap: int = 5,
    radius_decay: float = 0.79,
    momentum: float = 0.85,
) -> np.ndarray:
    """
    Generate interconnected 3D vascular tree structures via skeleton + distance transform.

    Grows branching trees with direction-based momentum (smooth curves),
    Murray's Law radius decay at bifurcations, and rasterizes using
    distance transform for smooth, non-lumpy tubes.

    Args:
        size: Volume size (cube)
        max_depth: Maximum branching depth
        branch_prob: Probability of branching at each eligible step
        min_branch_gap: Minimum steps between consecutive branches
        radius_decay: Radius multiplier at branch (0.79 ≈ Murray's Law)
        momentum: Direction smoothing (0=random walk, 1=straight line)

    Returns:
        Binary mask of shape (size, size, size)
    """
    from scipy.ndimage import distance_transform_edt

    segments = []  # List of (start, end, radius) tuples
    occupied = np.zeros((size, size, size), dtype=bool)  # Collision map
    init_radius = max(1.5, size / 30)
    step_len = max(2.0, size / 20)

    def _mark_occupied_3d(p_start, p_end, radius):
        """Rasterize a tube segment into the 3D occupied grid (half-radius zone)."""
        seg_len = np.linalg.norm(p_end - p_start)
        n_pts = max(2, int(seg_len * 2))
        mark_r = max(1.0, radius * 0.5)
        ri = int(np.ceil(mark_r))
        for t in np.linspace(0, 1, n_pts):
            pt = p_start + t * (p_end - p_start)
            iz = int(np.clip(pt[0], 0, size - 1))
            iy = int(np.clip(pt[1], 0, size - 1))
            ix = int(np.clip(pt[2], 0, size - 1))
            z0, z1 = max(0, iz - ri), min(size, iz + ri + 1)
            y0, y1 = max(0, iy - ri), min(size, iy + ri + 1)
            x0, x1 = max(0, ix - ri), min(size, ix + ri + 1)
            zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
            occupied[z0:z1, y0:y1, x0:x1][
                (zz - iz)**2 + (yy - iy)**2 + (xx - ix)**2 <= mark_r**2
            ] = True

    def _is_occupied_3d(pos):
        """Check if a position's centerline crosses an existing vessel."""
        iz = int(np.clip(pos[0], 0, size - 1))
        iy = int(np.clip(pos[1], 0, size - 1))
        ix = int(np.clip(pos[2], 0, size - 1))
        return occupied[iz, iy, ix]

    def _grow(pos, direction, radius, depth, max_steps):
        if depth > max_depth or radius < 0.5:
            return
        n_steps = np.random.randint(max(2, max_steps // 2), max(3, max_steps) + 1)
        steps_since_branch = min_branch_gap
        for s in range(n_steps):
            # Momentum-based direction update: smooth curves
            perturbation = np.random.randn(3) * 0.4
            direction = momentum * direction + (1 - momentum) * perturbation
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            new_pos = pos + direction * step_len

            # Reflect off boundaries
            for dim in range(3):
                if new_pos[dim] < 1 or new_pos[dim] >= size - 1:
                    direction[dim] = -direction[dim]
                    new_pos[dim] = np.clip(new_pos[dim], 1, size - 2)

            # Collision check: stop if centerline enters existing vessel interior
            if s > 2 and _is_occupied_3d(new_pos):
                break

            segments.append((pos.copy(), new_pos.copy(), radius))
            _mark_occupied_3d(pos, new_pos, radius)
            pos = new_pos.copy()
            steps_since_branch += 1

            # Branch (Murray's Law) — enforce minimum gap between branches
            if s > 2 and steps_since_branch >= min_branch_gap and np.random.random() < branch_prob:
                # Random perpendicular-ish branch direction
                rand_vec = np.random.randn(3)
                # Gram-Schmidt: remove component along current direction
                rand_vec = rand_vec - np.dot(rand_vec, direction) * direction
                norm = np.linalg.norm(rand_vec)
                if norm > 1e-6:
                    rand_vec = rand_vec / norm
                else:
                    rand_vec = np.random.randn(3)
                    rand_vec = rand_vec / (np.linalg.norm(rand_vec) + 1e-8)
                # Mix forward + perpendicular for branch angle
                branch_angle = np.random.uniform(0.4, 1.0)
                branch_dir = np.cos(branch_angle) * direction + np.sin(branch_angle) * rand_vec
                branch_dir = branch_dir / (np.linalg.norm(branch_dir) + 1e-8)
                _grow(pos, branch_dir, radius * radius_decay, depth + 1,
                      max(3, int(max_steps * 0.7)))
                steps_since_branch = 0

    # Single root, randomly placed — initial direction toward volume center
    pos = np.array([
        np.random.uniform(size * 0.2, size * 0.8),
        np.random.uniform(size * 0.2, size * 0.8),
        np.random.uniform(size * 0.2, size * 0.8),
    ])
    center = np.array([size / 2.0, size / 2.0, size / 2.0])
    d = center - pos
    d = d / (np.linalg.norm(d) + 1e-8)
    d = d + np.random.randn(3) * 0.2
    d = d / (np.linalg.norm(d) + 1e-8)
    _grow(pos, d, init_radius, 0, max_steps=25)

    if len(segments) == 0:
        return np.zeros((size, size, size), dtype=np.float64)

    # Rasterize: draw skeleton lines with per-segment radius
    skeleton = np.zeros((size, size, size), dtype=bool)
    radius_map = np.zeros((size, size, size), dtype=np.float64)

    for p_start, p_end, radius in segments:
        seg_len = np.linalg.norm(p_end - p_start)
        n_pts = max(2, int(seg_len * 2))
        for t in np.linspace(0, 1, n_pts):
            pt = p_start + t * (p_end - p_start)
            iz = int(np.clip(pt[0], 0, size - 1))
            iy = int(np.clip(pt[1], 0, size - 1))
            ix = int(np.clip(pt[2], 0, size - 1))
            skeleton[iz, iy, ix] = True
            radius_map[iz, iy, ix] = max(radius_map[iz, iy, ix], radius)

    if not skeleton.any():
        return np.zeros((size, size, size), dtype=np.float64)

    # Distance transform: for each voxel, get distance to nearest skeleton point
    # AND indices of that nearest skeleton point (to look up its radius)
    dist, nearest_idx = distance_transform_edt(~skeleton, return_distances=True,
                                                return_indices=True)

    # Look up the radius at the nearest skeleton point for each voxel
    nearest_radius = radius_map[nearest_idx[0], nearest_idx[1], nearest_idx[2]]

    # Mask: voxel is inside vessel if its distance to nearest skeleton <= that point's radius
    mask = (dist <= nearest_radius).astype(np.float64)

    return mask


def generate_nested_blobs_3d(
    size: int,
    n_tumors: int = 3,
    wt_size_range: Tuple[int, int] = (6, 16),
    tc_shrink: float = 0.65,
    et_shrink: float = 0.45,
) -> np.ndarray:
    """
    Generate hierarchical nested 3D blobs mimicking BraTS label structure.

    Produces (3, size, size, size) with channels [WT, TC, ET] where WT ⊃ TC ⊃ ET.
    Each tumor is a set of nested ellipsoids:
      - WT (whole tumor): outer shell + core + enhancing
      - TC (tumor core): core + enhancing (subset of WT)
      - ET (enhancing tumor): innermost region (subset of TC)

    Args:
        size: Volume size (cube)
        n_tumors: Number of independent tumor foci
        wt_size_range: Radius range for outermost (WT) ellipsoids
        tc_shrink: TC radius as fraction of WT radius
        et_shrink: ET radius as fraction of WT radius
    """
    wt = np.zeros((size, size, size), dtype=np.float32)
    tc = np.zeros((size, size, size), dtype=np.float32)
    et = np.zeros((size, size, size), dtype=np.float32)

    margin = wt_size_range[1] + 3
    z, y, x = np.ogrid[:size, :size, :size]

    for _ in range(n_tumors):
        cx = np.random.randint(margin, max(margin + 1, size - margin))
        cy = np.random.randint(margin, max(margin + 1, size - margin))
        cz = np.random.randint(margin, max(margin + 1, size - margin))

        # WT ellipsoid (largest)
        rx_wt = np.random.randint(*wt_size_range)
        ry_wt = np.random.randint(*wt_size_range)
        rz_wt = np.random.randint(*wt_size_range)
        wt_mask = (
            ((x - cx) / (rx_wt + 1e-6)) ** 2
            + ((y - cy) / (ry_wt + 1e-6)) ** 2
            + ((z - cz) / (rz_wt + 1e-6)) ** 2
        ) <= 1

        # TC ellipsoid (subset of WT) — slightly offset center for realism
        offset = np.random.uniform(-2, 2, 3)
        rx_tc = max(2, int(rx_wt * tc_shrink))
        ry_tc = max(2, int(ry_wt * tc_shrink))
        rz_tc = max(2, int(rz_wt * tc_shrink))
        tc_mask = (
            ((x - cx - offset[0]) / (rx_tc + 1e-6)) ** 2
            + ((y - cy - offset[1]) / (ry_tc + 1e-6)) ** 2
            + ((z - cz - offset[2]) / (rz_tc + 1e-6)) ** 2
        ) <= 1

        # ET ellipsoid (subset of TC)
        rx_et = max(1, int(rx_wt * et_shrink))
        ry_et = max(1, int(ry_wt * et_shrink))
        rz_et = max(1, int(rz_wt * et_shrink))
        et_mask = (
            ((x - cx - offset[0]) / (rx_et + 1e-6)) ** 2
            + ((y - cy - offset[1]) / (ry_et + 1e-6)) ** 2
            + ((z - cz - offset[2]) / (rz_et + 1e-6)) ** 2
        ) <= 1

        wt = np.maximum(wt, wt_mask.astype(np.float32))
        tc = np.maximum(tc, (tc_mask & wt_mask).astype(np.float32))
        et = np.maximum(et, (et_mask & tc_mask & wt_mask).astype(np.float32))

    return np.stack([wt, tc, et], axis=0)  # (3, D, H, W)


def generate_vessel_tumor_combo_3d(
    size: int,
    n_tumors: int = 3,
    tumor_size_range: Tuple[int, int] = (2, 8),
) -> np.ndarray:
    """
    Generate combined vessel + tumor mask for hepatic vessel segmentation.

    Channel 0: Branching vessel tree (frangi generator)
    Channel 1: Random blob tumors (gaussian blobs)

    Returns (2, size, size, size) with channels [vessel, tumour].
    """
    vessel = generate_frangi_vessels_3d(size).astype(np.float32)
    tumour = generate_gaussian_blobs_3d(
        size, n_blobs=n_tumors, size_range=tumor_size_range,
    ).astype(np.float32)
    return np.stack([vessel, tumour], axis=0)  # (2, D, H, W)


def get_mask_generator_3d(strategy: str) -> Callable:
    """Get 3D mask generator function by name."""
    generators = {
        "gaussian_blobs": generate_gaussian_blobs_3d,
        "random_shapes": generate_random_shapes_3d,
        "pink_threshold": generate_pink_threshold_3d,
        "voronoi": generate_voronoi_cells_3d,
        "frangi": generate_frangi_vessels_3d,
        "nested_blobs": generate_nested_blobs_3d,
        "vessel_tumor_combo": generate_vessel_tumor_combo_3d,
    }
    if strategy not in generators:
        raise ValueError(f"Unknown 3D mask strategy: {strategy}. Available: {list(generators.keys())}")
    return generators[strategy]


# =============================================================================
# SYNTHETIC 3D DATASET
# =============================================================================

def composite_image_with_mask_3d(
    background: np.ndarray,
    mask: np.ndarray,
    fg_intensity_shift: float = 0.3,
    fg_noise_scale: float = 0.15,
    edge_width: int = 1,
    edge_brightness: float = 0.2,
) -> np.ndarray:
    """
    Make masked regions visually distinct in a 3D synthetic image.

    Applies intensity shift, additional noise, and edge highlighting
    inside the masked region so the data looks like a real segmentation task.

    Args:
        background: 3D noise array, normalized [0, 1]
        mask: binary 3D mask (same shape as background)
        fg_intensity_shift: intensity offset for foreground
        fg_noise_scale: scale of additional foreground noise
        edge_width: dilation iterations for edge detection
        edge_brightness: intensity boost at edges

    Returns:
        Composited image, clipped to [0, 1]
    """
    result = background.copy()
    mask_bool = mask > 0.5

    # 1. Intensity shift in foreground
    result[mask_bool] += fg_intensity_shift

    # 2. Add different noise inside foreground
    fg_noise = np.random.randn(*background.shape) * fg_noise_scale
    result[mask_bool] += fg_noise[mask_bool]

    # 3. Edge highlighting
    dilated = ndimage.binary_dilation(mask_bool, iterations=edge_width)
    eroded = ndimage.binary_erosion(mask_bool, iterations=max(1, edge_width))
    edge = dilated & ~eroded
    result[edge] += edge_brightness

    return np.clip(result, 0, 1)


class SyntheticSegmentation3DDataset(Dataset):
    """
    3D synthetic dataset for preconditioner estimation.

    On-the-fly generation with deterministic per-index seeding.
    Supports all noise types and mask strategies from 2D, ported to 3D.
    Foreground regions are visually distinct from background.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        patch_size: int = 64,
        in_channels: int = 4,
        num_classes: int = 1,
        noise_type: str = "pink",
        mask_strategy: str = "gaussian_blobs",
        seed: int = 42,
        realistic: bool = True,
    ):
        self.n_samples = n_samples
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.noise_type = noise_type
        self.mask_strategy = mask_strategy
        self.seed = seed
        self.realistic = realistic

        self._noise_fn = get_noise_generator_3d(noise_type)
        self._mask_fn = get_mask_generator_3d(mask_strategy)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Deterministic per-index seed
        rng_state = np.random.get_state()
        np.random.seed(self.seed + idx)

        # Generate mask(s)
        raw_mask = self._mask_fn(self.patch_size)

        if raw_mask.ndim == 4:
            # Generator returned multi-channel directly (e.g. nested_blobs, vessel_tumor_combo)
            mask_multi = raw_mask.astype(np.float32)
            mask_composite = (mask_multi.max(axis=0) > 0.5).astype(np.float32)
        elif self.num_classes > 1:
            # Single-channel generator, stack independent masks for each class
            mask_channels = [raw_mask.astype(np.float32)]
            for _ in range(self.num_classes - 1):
                m = self._mask_fn(self.patch_size).astype(np.float32)
                mask_channels.append(m)
            mask_multi = np.stack(mask_channels, axis=0)  # (C_out, D, H, W)
            mask_composite = (mask_multi.max(axis=0) > 0.5).astype(np.float32)
        else:
            mask_composite = raw_mask.astype(np.float32)
            mask_multi = None

        # Generate multi-channel image with foreground distinction
        channels = []
        for _ in range(self.in_channels):
            noise = self._noise_fn(self.patch_size)
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

            if self.realistic:
                shift = np.random.uniform(0.15, 0.4) * np.random.choice([-1, 1])
                noise = composite_image_with_mask_3d(
                    noise, mask_composite,
                    fg_intensity_shift=shift,
                    fg_noise_scale=0.1,
                    edge_width=max(1, self.patch_size // 32),
                    edge_brightness=0.15,
                )

            channels.append(noise)
        image = np.stack(channels, axis=0).astype(np.float32)  # (C_in, D, H, W)

        # Restore RNG state
        np.random.set_state(rng_state)

        image_t = torch.from_numpy(image).float()
        if mask_multi is not None:
            mask_t = torch.from_numpy(mask_multi).float()  # (C_out, D, H, W)
        else:
            mask_t = torch.from_numpy(mask_composite).float().unsqueeze(0)  # (1, D, H, W)

        return image_t, mask_t


# =============================================================================
# UNIFIED FACTORY
# =============================================================================

def create_synthetic_segmentation_dataset(
    spatial_dims: int = 2,
    **kwargs,
) -> Dataset:
    """
    Factory that auto-selects 2D or 3D synthetic segmentation dataset.

    Args:
        spatial_dims: 2 for 2D, 3 for 3D. Auto-detected from context.
        **kwargs: Passed to the dataset constructor.
            For 2D: n_samples, image_size, in_channels, noise_type, mask_strategy, seed
            For 3D: n_samples, patch_size, in_channels, noise_type, mask_strategy, seed

    Returns:
        SyntheticSegmentationDataset (2D) or SyntheticSegmentation3DDataset (3D)
    """
    if spatial_dims == 3:
        # Remap image_size -> patch_size if needed (unified interface)
        if "image_size" in kwargs and "patch_size" not in kwargs:
            kwargs["patch_size"] = kwargs.pop("image_size")
        elif "image_size" in kwargs:
            kwargs.pop("image_size")
        return SyntheticSegmentation3DDataset(**kwargs)
    else:
        from .synthetic import SyntheticSegmentationDataset
        # Remap patch_size -> image_size if needed
        if "patch_size" in kwargs and "image_size" not in kwargs:
            kwargs["image_size"] = kwargs.pop("patch_size")
        elif "patch_size" in kwargs:
            kwargs.pop("patch_size")
        return SyntheticSegmentationDataset(**kwargs)


if __name__ == "__main__":
    print("Testing 3D noise generators...")
    for name in ["white", "pink", "brown"]:
        fn = get_noise_generator_3d(name)
        noise = fn(32)
        print(f"  {name}: shape={noise.shape}, mean={noise.mean():.3f}, std={noise.std():.3f}")

    print("\nTesting 3D mask generators...")
    for name in ["gaussian_blobs", "random_shapes", "pink_threshold", "voronoi", "frangi"]:
        fn = get_mask_generator_3d(name)
        mask = fn(32)
        print(f"  {name}: shape={mask.shape}, coverage={mask.mean()*100:.1f}%")

    print("\nTesting SyntheticSegmentation3DDataset with all mask strategies...")
    for strat in ["gaussian_blobs", "random_shapes", "pink_threshold", "voronoi", "frangi"]:
        dataset = SyntheticSegmentation3DDataset(n_samples=2, patch_size=32, mask_strategy=strat)
        image, mask = dataset[0]
        print(f"  {strat}: Image={image.shape}, Mask={mask.shape}, coverage={mask.mean()*100:.1f}%")

    print("\nTesting unified factory...")
    ds3d = create_synthetic_segmentation_dataset(spatial_dims=3, n_samples=2, patch_size=32)
    img, msk = ds3d[0]
    print(f"  3D: {img.shape}, {msk.shape}")

    print("\n3D synthetic data generation working!")

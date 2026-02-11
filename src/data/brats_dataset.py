"""
BraTS 2025 Dataset for 3D brain tumor segmentation.

Loads NIfTI volumes, extracts random 3D patches, with per-modality z-score
normalization. Supports whole tumor (WT), tumor core (TC), and enhancing
tumor (ET) targets.

Data path: /eos/project/d/diagbox/BRATS2025/.../BraTS2025-GLI-PRE-Challenge-TrainingData/
Volumes: (182, 218, 182), 4 MRI modalities (T1c, T1n, T2f, T2w), isotropic 1mm.
Seg labels: 0=bg, 1=NCR, 2=ED, 3=ET.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


BRATS_DEFAULT_ROOT = "/eos/project/d/diagbox/BRATS2025/Glioma_seg_pre_post_treatment_mri/PRE/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS2025-GLI-PRE-Challenge-TrainingData"

MODALITIES = ["t1c", "t1n", "t2f", "t2w"]


def _discover_cases(root: str):
    """Discover all BraTS case directories sorted by name."""
    root = Path(root)
    cases = sorted([d for d in root.iterdir() if d.is_dir()])
    return cases


def _load_volume(case_path):
    """
    Load and normalize a BraTS volume.

    Returns:
        image: np.ndarray of shape (4, D, H, W), float32, z-score normalized
        seg: np.ndarray of shape (D, H, W), int labels
    """
    case_path = Path(case_path)
    case_name = case_path.name

    # Load all modalities
    channels = []
    for mod in MODALITIES:
        nii_path = case_path / f"{case_name}-{mod}.nii.gz"
        vol = nib.load(str(nii_path)).get_fdata().astype(np.float32)

        # Per-modality z-score normalization (brain mask only)
        brain_mask = vol > 0
        if brain_mask.sum() > 0:
            mean = vol[brain_mask].mean()
            std = vol[brain_mask].std()
            vol = np.where(brain_mask, (vol - mean) / (std + 1e-8), 0.0)

        channels.append(vol)

    image = np.stack(channels, axis=0)  # (4, D, H, W)

    # Load segmentation
    seg_path = case_path / f"{case_name}-seg.nii.gz"
    seg = nib.load(str(seg_path)).get_fdata().astype(np.int64)

    return image, seg


def _binarize_target(seg: np.ndarray, target: str) -> np.ndarray:
    """Convert segmentation labels to the specified target.

    For 'multilabel', returns (3, D, H, W) with channels [WT, TC, ET].
    For single targets, returns (D, H, W) binary mask.
    """
    if target == "multilabel":
        wt = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.float32)
        tc = ((seg == 1) | (seg == 3)).astype(np.float32)
        et = (seg == 3).astype(np.float32)
        return np.stack([wt, tc, et], axis=0)  # (3, D, H, W)
    elif target == "wt":
        return ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.float32)
    elif target == "tc":
        return ((seg == 1) | (seg == 3)).astype(np.float32)
    elif target == "et":
        return (seg == 3).astype(np.float32)
    else:
        raise ValueError(f"Unknown target: {target}. Use 'multilabel', 'wt', 'tc', or 'et'.")


class BraTSPatchDataset(Dataset):
    """
    BraTS 3D patch dataset.

    Extracts random 3D patches from volumes. 50% tumor-centered,
    50% random foreground. Includes random flips and intensity scaling.

    Uses a per-instance dict cache that is shared across workers via
    fork (copy-on-write) when prefilled, or populated per-worker otherwise.
    """

    def __init__(
        self,
        root: str = BRATS_DEFAULT_ROOT,
        split: str = "train",
        patch_size: int = 128,
        patches_per_volume: int = 4,
        target: str = "wt",
        augment: bool = True,
        seed: int = 42,
        cache_volumes: int = 8,
    ):
        self.root = Path(root)
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.target = target
        self.augment = augment and (split == "train")
        self._cache_max = cache_volumes

        # Discover and split cases (85/15)
        all_cases = _discover_cases(root)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_cases))
        split_idx = int(0.85 * len(all_cases))

        if split == "train":
            self.cases = [all_cases[i] for i in indices[:split_idx]]
        else:
            self.cases = [all_cases[i] for i in indices[split_idx:]]

        # Instance-level cache: {case_path_str: (image, mask, tumor_voxels)}
        self._cache = {}
        # Precompute tumor voxel indices per volume to avoid np.argwhere in __getitem__
        self._tumor_cache = {}

    def _get_volume(self, case_path):
        """Load volume with instance-level LRU cache."""
        key = str(case_path)
        if key in self._cache:
            return self._cache[key]

        image, seg = _load_volume(case_path)
        mask = _binarize_target(seg, self.target)
        tumor_voxels = np.argwhere(mask > 0)

        # Evict oldest if cache full
        if len(self._cache) >= self._cache_max:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            del self._tumor_cache[oldest_key]

        self._cache[key] = (image, mask)
        self._tumor_cache[key] = tumor_voxels
        return image, mask

    def __len__(self):
        return len(self.cases) * self.patches_per_volume

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        vol_idx = idx // self.patches_per_volume
        case_path = self.cases[vol_idx]

        image, mask = self._get_volume(case_path)
        tumor_voxels = self._tumor_cache[str(case_path)]

        # Extract patch
        patch_img, patch_mask = self._extract_patch(image, mask, tumor_voxels, idx)

        # Augmentation
        if self.augment:
            patch_img, patch_mask = self._augment(patch_img, patch_mask, idx)

        image_t = torch.from_numpy(np.ascontiguousarray(patch_img)).float()
        mask_arr = np.ascontiguousarray(patch_mask)
        if mask_arr.ndim == 3:
            # Single-channel: add channel dim
            mask_t = torch.from_numpy(mask_arr).float().unsqueeze(0)
        else:
            # Multi-channel (C, D, H, W): already has channel dim
            mask_t = torch.from_numpy(mask_arr).float()

        return image_t, mask_t

    def _extract_patch(self, image, mask, tumor_voxels, idx):
        """Extract a random 3D patch. 50% tumor-centered, 50% random foreground."""
        _, D, H, W = image.shape
        ps = self.patch_size
        rng = np.random.RandomState(idx)

        use_tumor = (rng.random() < 0.5) and (len(tumor_voxels) > 0)

        if use_tumor:
            center = tumor_voxels[rng.randint(len(tumor_voxels))]
            d0 = int(np.clip(center[0] - ps // 2, 0, max(0, D - ps)))
            h0 = int(np.clip(center[1] - ps // 2, 0, max(0, H - ps)))
            w0 = int(np.clip(center[2] - ps // 2, 0, max(0, W - ps)))
        else:
            d0 = rng.randint(0, max(1, D - ps))
            h0 = rng.randint(0, max(1, H - ps))
            w0 = rng.randint(0, max(1, W - ps))

        patch_img = image[:, d0:d0+ps, h0:h0+ps, w0:w0+ps]
        if mask.ndim == 4:
            # Multi-channel: (C, D, H, W)
            patch_mask = mask[:, d0:d0+ps, h0:h0+ps, w0:w0+ps]
        else:
            patch_mask = mask[d0:d0+ps, h0:h0+ps, w0:w0+ps]

        # Pad if volume is smaller than patch_size
        _, pd, ph, pw = patch_img.shape
        if pd < ps or ph < ps or pw < ps:
            padded_img = np.zeros((4, ps, ps, ps), dtype=patch_img.dtype)
            if patch_mask.ndim == 4:
                C = patch_mask.shape[0]
                padded_mask = np.zeros((C, ps, ps, ps), dtype=patch_mask.dtype)
                padded_mask[:, :pd, :ph, :pw] = patch_mask
            else:
                padded_mask = np.zeros((ps, ps, ps), dtype=patch_mask.dtype)
                padded_mask[:pd, :ph, :pw] = patch_mask
            padded_img[:, :pd, :ph, :pw] = patch_img
            patch_img = padded_img
            patch_mask = padded_mask

        return patch_img, patch_mask

    def _augment(self, image, mask, idx):
        """Random 3D axis flips + intensity scaling."""
        rng = np.random.RandomState(idx * 7 + 13)

        # For multi-channel mask (C,D,H,W), spatial axes are 1,2,3 same as image
        mask_offset = 0 if mask.ndim == 4 else 1
        for axis in [1, 2, 3]:
            if rng.random() < 0.5:
                image = np.flip(image, axis=axis)
                mask = np.flip(mask, axis=axis - mask_offset)

        for c in range(image.shape[0]):
            scale = rng.uniform(0.9, 1.1)
            image[c] = image[c] * scale

        return image, mask

    def get_full_volume(self, case_idx: int):
        """Load a full volume for sliding window inference."""
        case_path = self.cases[case_idx]
        image, seg = _load_volume(case_path)
        mask = _binarize_target(seg, self.target)
        mask_t = torch.from_numpy(mask.copy()).float()
        if mask_t.ndim == 3:
            mask_t = mask_t.unsqueeze(0)
        return torch.from_numpy(image.copy()).float(), mask_t


def get_brats_dataloaders(
    root: str = BRATS_DEFAULT_ROOT,
    patch_size: int = 128,
    patches_per_volume: int = 4,
    target: str = "wt",
    batch_size: int = 2,
    num_workers: int = 4,
):
    """Create train and val dataloaders for BraTS."""
    train_dataset = BraTSPatchDataset(
        root=root, split="train", patch_size=patch_size,
        patches_per_volume=patches_per_volume, target=target, augment=True,
    )
    val_dataset = BraTSPatchDataset(
        root=root, split="val", patch_size=patch_size,
        patches_per_volume=patches_per_volume, target=target, augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing BraTS Dataset...")
    dataset = BraTSPatchDataset(patch_size=64, patches_per_volume=2)
    print(f"Dataset size: {len(dataset)}")

    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")

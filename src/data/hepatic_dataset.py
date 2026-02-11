"""
Hepatic Vessel Segmentation Dataset (Medical Segmentation Decathlon Task08).

303 training CT volumes, (512, 512, ~40-70 slices), anisotropic ~0.9x0.9x5mm.
Labels: 0=background, 1=vessel, 2=tumour.
Binary target: "vessel" (label == 1), "tumour" (label == 2), "all" (label > 0).

3D patch-based loading, mirrors BraTS dataset design.
"""

import os
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


HEPATIC_DEFAULT_ROOT = "./data/Task08_HepaticVessel"


def _discover_hepatic_cases(root: str):
    """Discover all hepatic vessel training volumes (image + label pairs)."""
    root = Path(root)
    images_dir = root / "imagesTr"
    labels_dir = root / "labelsTr"

    cases = []
    for img_path in sorted(images_dir.glob("hepaticvessel_*.nii.gz")):
        label_path = labels_dir / img_path.name
        if label_path.exists():
            cases.append((img_path, label_path))
    return cases


def _load_hepatic_volume(img_path, label_path):
    """
    Load and normalize a hepatic vessel CT volume.

    Returns:
        image: np.ndarray of shape (1, H, W, D), float32, clipped + z-score
        seg: np.ndarray of shape (H, W, D), int labels
    """
    vol = nib.load(str(img_path)).get_fdata().astype(np.float32)
    seg = nib.load(str(label_path)).get_fdata().astype(np.int64)

    # CT window: clip to [-200, 300] HU (liver window), then z-score
    vol = np.clip(vol, -200, 300)
    mean, std = vol.mean(), vol.std()
    vol = (vol - mean) / (std + 1e-8)

    image = vol[np.newaxis]  # (1, H, W, D)
    return image, seg


def _binarize_hepatic_target(seg: np.ndarray, target: str) -> np.ndarray:
    """Convert segmentation labels to the specified target.

    For 'multilabel', returns (2, H, W, D) with channels [vessel, tumour].
    For single targets, returns (H, W, D) binary mask.
    """
    if target == "multilabel":
        vessel = (seg == 1).astype(np.float32)
        tumour = (seg == 2).astype(np.float32)
        return np.stack([vessel, tumour], axis=0)  # (2, H, W, D)
    elif target == "vessel":
        return (seg == 1).astype(np.float32)
    elif target == "tumour":
        return (seg == 2).astype(np.float32)
    elif target == "all":
        return (seg > 0).astype(np.float32)
    else:
        raise ValueError(f"Unknown target: {target}. Use 'multilabel', 'vessel', 'tumour', or 'all'.")


class HepaticVesselDataset(Dataset):
    """
    Hepatic Vessel 3D patch dataset.

    Extracts random 3D patches from CT volumes. 50% target-centered,
    50% random. Includes random flips and intensity scaling.
    """

    def __init__(
        self,
        root: str = HEPATIC_DEFAULT_ROOT,
        split: str = "train",
        patch_size: int = 64,
        patches_per_volume: int = 8,
        target: str = "vessel",
        augment: bool = True,
        seed: int = 42,
        cache_volumes: int = 8,
    ):
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.target = target
        self.augment = augment and (split == "train")
        self._cache_max = cache_volumes

        all_cases = _discover_hepatic_cases(root)
        if len(all_cases) == 0:
            raise FileNotFoundError(
                f"No hepatic vessel volumes found in {root}. "
                f"Expected imagesTr/ and labelsTr/ subdirectories."
            )

        # 85/15 train/val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_cases))
        split_idx = int(0.85 * len(all_cases))

        if split == "train":
            self.cases = [all_cases[i] for i in indices[:split_idx]]
        else:
            self.cases = [all_cases[i] for i in indices[split_idx:]]

        self._cache = {}
        self._target_cache = {}

    def _get_volume(self, img_path, label_path):
        """Load volume with instance-level LRU cache."""
        key = str(img_path)
        if key in self._cache:
            return self._cache[key]

        image, seg = _load_hepatic_volume(img_path, label_path)
        mask = _binarize_hepatic_target(seg, self.target)
        # For multilabel (C, H, W, D), check any channel; for single (H, W, D) check directly
        if mask.ndim == 4:
            target_voxels = np.argwhere(mask.max(axis=0) > 0)
        else:
            target_voxels = np.argwhere(mask > 0)

        if len(self._cache) >= self._cache_max:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            del self._target_cache[oldest_key]

        self._cache[key] = (image, mask)
        self._target_cache[key] = target_voxels
        return image, mask

    def __len__(self):
        return len(self.cases) * self.patches_per_volume

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        vol_idx = idx // self.patches_per_volume
        img_path, label_path = self.cases[vol_idx]

        image, mask = self._get_volume(img_path, label_path)
        target_voxels = self._target_cache[str(img_path)]

        patch_img, patch_mask = self._extract_patch(image, mask, target_voxels, idx)

        if self.augment:
            patch_img, patch_mask = self._augment(patch_img, patch_mask, idx)

        image_t = torch.from_numpy(np.ascontiguousarray(patch_img)).float()
        mask_arr = np.ascontiguousarray(patch_mask)
        if mask_arr.ndim == 3:
            # Single-channel: add channel dim
            mask_t = torch.from_numpy(mask_arr).float().unsqueeze(0)
        else:
            # Multi-channel (C, H, W, D): already has channel dim
            mask_t = torch.from_numpy(mask_arr).float()

        return image_t, mask_t

    def _extract_patch(self, image, mask, target_voxels, idx):
        """Extract a random 3D patch. 50% target-centered, 50% random."""
        _, H, W, D = image.shape
        ps = self.patch_size
        rng = np.random.RandomState(idx)

        use_target = (rng.random() < 0.5) and (len(target_voxels) > 0)

        if use_target:
            center = target_voxels[rng.randint(len(target_voxels))]
            h0 = int(np.clip(center[0] - ps // 2, 0, max(0, H - ps)))
            w0 = int(np.clip(center[1] - ps // 2, 0, max(0, W - ps)))
            d0 = int(np.clip(center[2] - ps // 2, 0, max(0, D - ps)))
        else:
            h0 = rng.randint(0, max(1, H - ps))
            w0 = rng.randint(0, max(1, W - ps))
            d0 = rng.randint(0, max(1, D - ps))

        patch_img = image[:, h0:h0+ps, w0:w0+ps, d0:d0+ps]
        if mask.ndim == 4:
            # Multi-channel: (C, H, W, D)
            patch_mask = mask[:, h0:h0+ps, w0:w0+ps, d0:d0+ps]
        else:
            patch_mask = mask[h0:h0+ps, w0:w0+ps, d0:d0+ps]

        # Pad if volume dimension < patch_size
        _, ph, pw, pd = patch_img.shape
        if ph < ps or pw < ps or pd < ps:
            padded_img = np.zeros((1, ps, ps, ps), dtype=patch_img.dtype)
            if patch_mask.ndim == 4:
                C = patch_mask.shape[0]
                padded_mask = np.zeros((C, ps, ps, ps), dtype=patch_mask.dtype)
                padded_mask[:, :ph, :pw, :pd] = patch_mask
            else:
                padded_mask = np.zeros((ps, ps, ps), dtype=patch_mask.dtype)
                padded_mask[:ph, :pw, :pd] = patch_mask
            padded_img[:, :ph, :pw, :pd] = patch_img
            patch_img = padded_img
            patch_mask = padded_mask

        return patch_img, patch_mask

    def _augment(self, image, mask, idx):
        """Random 3D axis flips + intensity scaling."""
        rng = np.random.RandomState(idx * 7 + 13)

        # For multi-channel mask (C,H,W,D), spatial axes are 1,2,3 same as image
        mask_offset = 0 if mask.ndim == 4 else 1
        for axis in [1, 2, 3]:
            if rng.random() < 0.5:
                image = np.flip(image, axis=axis)
                mask = np.flip(mask, axis=axis - mask_offset)

        scale = rng.uniform(0.9, 1.1)
        image = image * scale

        return image, mask

    def get_full_volume(self, case_idx: int):
        """Load a full volume for sliding window inference."""
        img_path, label_path = self.cases[case_idx]
        image, seg = _load_hepatic_volume(img_path, label_path)
        mask = _binarize_hepatic_target(seg, self.target)
        mask_t = torch.from_numpy(mask.copy()).float()
        if mask_t.ndim == 3:
            mask_t = mask_t.unsqueeze(0)
        return torch.from_numpy(image.copy()).float(), mask_t


def get_hepatic_dataloaders(
    root: str = HEPATIC_DEFAULT_ROOT,
    patch_size: int = 64,
    patches_per_volume: int = 8,
    target: str = "vessel",
    batch_size: int = 2,
    num_workers: int = 4,
):
    """Create train and val dataloaders for Hepatic Vessel."""
    train_dataset = HepaticVesselDataset(
        root=root, split="train", patch_size=patch_size,
        patches_per_volume=patches_per_volume, target=target, augment=True,
    )
    val_dataset = HepaticVesselDataset(
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
    print("Testing Hepatic Vessel Dataset...")
    dataset = HepaticVesselDataset(patch_size=64, patches_per_volume=2)
    print(f"Dataset size: {len(dataset)}")

    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")

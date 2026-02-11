"""
PyTorch Dataset classes for retinal vessel segmentation.

Provides dataset classes for DRIVE and STARE with support for:
- Standard training/validation splits
- Patch-based training (for memory efficiency with DP-SGD)
- Data augmentation via albumentations
- Combined multi-dataset training
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentations(
    image_size: int = 512,
    p_flip: float = 0.5,
    p_rotate: float = 0.5,
    p_elastic: float = 0.3,
) -> A.Compose:
    """Get augmentation pipeline for training."""
    return A.Compose([
        A.HorizontalFlip(p=p_flip),
        A.VerticalFlip(p=p_flip),
        A.RandomRotate90(p=p_rotate),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            border_mode=0,
            p=0.5
        ),
        A.ElasticTransform(
            alpha=120,
            sigma=6,
            p=p_elastic
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.GaussNoise(p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_validation_augmentations() -> A.Compose:
    """Get augmentation pipeline for validation (only normalization)."""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


class BaseVesselDataset(Dataset):
    """Base class for retinal vessel segmentation datasets."""

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        fov_dir: Optional[str] = None,
        transform: Optional[A.Compose] = None,
        image_ids: Optional[List[str]] = None,
    ):
        """
        Args:
            images_dir: Directory containing preprocessed images
            masks_dir: Directory containing vessel masks
            fov_dir: Directory containing FOV masks (optional)
            transform: Albumentations transform
            image_ids: Specific image IDs to use (for train/val splits)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.fov_dir = Path(fov_dir) if fov_dir else None
        self.transform = transform

        # Get list of image files
        if image_ids is not None:
            self.image_files = [self.images_dir / f"{id_}.png" for id_ in image_ids]
        else:
            self.image_files = sorted(self.images_dir.glob("*.png"))

        # Verify all files exist
        self.image_files = [f for f in self.image_files if f.exists()]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_files[idx]
        img_id = img_path.stem

        # Load image
        image = np.array(Image.open(img_path))

        # Load mask
        mask_path = self.masks_dir / f"{img_id}.png"
        if mask_path.exists():
            mask = np.array(Image.open(mask_path))
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32)

        # Load FOV mask if available
        if self.fov_dir:
            fov_path = self.fov_dir / f"{img_id}.png"
            if fov_path.exists():
                fov = np.array(Image.open(fov_path))
                fov = (fov > 127).astype(np.float32)
            else:
                fov = np.ones(image.shape[:2], dtype=np.float32)
        else:
            fov = np.ones(image.shape[:2], dtype=np.float32)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, masks=[mask, fov])
            image = transformed["image"]
            mask, fov = transformed["masks"]
            mask = mask.unsqueeze(0) if len(mask.shape) == 2 else mask
            fov = fov.unsqueeze(0) if len(fov.shape) == 2 else fov
        else:
            # Convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            fov = torch.from_numpy(fov).unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "fov": fov,
            "id": img_id,
        }


class DRIVEDataset(BaseVesselDataset):
    """
    DRIVE dataset for retinal vessel segmentation.

    The DRIVE dataset contains:
    - 40 images total (20 training, 20 test)
    - Image resolution: 565 x 584
    - Manual annotations by 2 experts (1st_manual used as ground truth)

    Standard split: images 21-40 for training, 01-20 for testing
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "training",
        transform: Optional[A.Compose] = None,
        validation_fold: Optional[int] = None,
        n_folds: int = 5,
        train: bool = True,
    ):
        """
        Args:
            data_dir: Path to preprocessed DRIVE data
            split: "training" or "test"
            transform: Albumentations transform
            validation_fold: If set, creates a k-fold split for training set
            n_folds: Number of folds for cross-validation
            train: If True and validation_fold is set, returns train split; else val split
        """
        data_path = Path(data_dir)
        images_dir = data_path / split / "images"
        masks_dir = data_path / split / "masks"
        fov_dir = data_path / split / "fov"

        # Get all image IDs
        all_ids = [f.stem for f in sorted(images_dir.glob("*.png"))]

        # Apply k-fold split if requested (for training set only)
        if validation_fold is not None and split == "training":
            np.random.seed(42)
            indices = np.random.permutation(len(all_ids))
            fold_size = len(all_ids) // n_folds
            val_start = validation_fold * fold_size
            val_end = val_start + fold_size

            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

            if train:
                image_ids = [all_ids[i] for i in train_indices]
            else:
                image_ids = [all_ids[i] for i in val_indices]
        else:
            image_ids = all_ids

        super().__init__(
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            fov_dir=str(fov_dir) if fov_dir.exists() else None,
            transform=transform,
            image_ids=image_ids,
        )

        self.split = split
        self.dataset_name = "DRIVE"


class STAREDataset(BaseVesselDataset):
    """
    STARE dataset for retinal vessel segmentation.

    The STARE dataset contains:
    - 20 images total
    - Image resolution: 700 x 605
    - Manual annotations by 2 experts (Adam Hoover and Valentina Kouznetsova)

    No standard train/test split; typically use leave-one-out or k-fold.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[A.Compose] = None,
        image_ids: Optional[List[str]] = None,
        validation_fold: Optional[int] = None,
        n_folds: int = 5,
        train: bool = True,
    ):
        """
        Args:
            data_dir: Path to preprocessed STARE data
            transform: Albumentations transform
            image_ids: Specific image IDs to use
            validation_fold: Fold index for cross-validation
            n_folds: Number of folds
            train: If True and validation_fold is set, returns train split; else val split
        """
        data_path = Path(data_dir)
        images_dir = data_path / "images"
        masks_dir = data_path / "masks"
        fov_dir = data_path / "fov"

        # Get all image IDs
        if image_ids is None:
            all_ids = [f.stem for f in sorted(images_dir.glob("*.png"))]
        else:
            all_ids = image_ids

        # Apply k-fold split if requested
        if validation_fold is not None:
            np.random.seed(42)
            indices = np.random.permutation(len(all_ids))
            fold_size = len(all_ids) // n_folds
            val_start = validation_fold * fold_size
            val_end = val_start + fold_size

            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

            if train:
                image_ids = [all_ids[i] for i in train_indices]
            else:
                image_ids = [all_ids[i] for i in val_indices]
        else:
            image_ids = all_ids

        super().__init__(
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            fov_dir=str(fov_dir) if fov_dir.exists() else None,
            transform=transform,
            image_ids=image_ids,
        )

        self.dataset_name = "STARE"


class HRFDataset(BaseVesselDataset):
    """
    HRF (High Resolution Fundus) dataset for retinal vessel segmentation.

    The HRF dataset contains:
    - 45 images total (15 healthy, 15 diabetic retinopathy, 15 glaucoma)
    - Image resolution: 3504 x 2336 (very high resolution)
    - Manual annotations and FOV masks provided

    No standard train/test split; typically use k-fold cross-validation.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[A.Compose] = None,
        validation_fold: Optional[int] = None,
        n_folds: int = 5,
        train: bool = True,
    ):
        """
        Args:
            data_dir: Path to preprocessed HRF data
            transform: Albumentations transform
            validation_fold: Fold index for cross-validation
            n_folds: Number of folds
            train: If True and validation_fold is set, returns train split; else val split
        """
        data_path = Path(data_dir)
        images_dir = data_path / "images"
        masks_dir = data_path / "masks"
        fov_dir = data_path / "fov"

        # Get all image IDs
        all_ids = [f.stem for f in sorted(images_dir.glob("*.png"))]

        # Apply k-fold split if requested
        if validation_fold is not None:
            np.random.seed(42)
            indices = np.random.permutation(len(all_ids))
            fold_size = len(all_ids) // n_folds
            val_start = validation_fold * fold_size
            val_end = val_start + fold_size

            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

            if train:
                image_ids = [all_ids[i] for i in train_indices]
            else:
                image_ids = [all_ids[i] for i in val_indices]
        else:
            image_ids = all_ids

        super().__init__(
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            fov_dir=str(fov_dir) if fov_dir.exists() else None,
            transform=transform,
            image_ids=image_ids,
        )

        self.dataset_name = "HRF"


class PatchDataset(Dataset):
    """
    Extracts random patches from images for memory-efficient training.

    Useful for DP-SGD which requires per-sample gradient computation,
    making large batch sizes memory-intensive.
    """

    def __init__(
        self,
        base_dataset: BaseVesselDataset,
        patch_size: int = 128,
        patches_per_image: int = 10,
        transform: Optional[A.Compose] = None,
    ):
        """
        Args:
            base_dataset: Base dataset to extract patches from
            patch_size: Size of extracted patches
            patches_per_image: Number of patches to extract per image
            transform: Additional transforms for patches
        """
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.transform = transform

        # Pre-compute patch locations (can be regenerated each epoch)
        self._generate_patches()

    def _generate_patches(self):
        """Generate random patch locations for all images."""
        self.patches = []

        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset[idx]
            h, w = sample["mask"].shape[-2:]

            for _ in range(self.patches_per_image):
                y = np.random.randint(0, max(1, h - self.patch_size))
                x = np.random.randint(0, max(1, w - self.patch_size))
                self.patches.append((idx, y, x))

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_idx, y, x = self.patches[idx]
        sample = self.base_dataset[img_idx]

        # Extract patch
        image = sample["image"][:, y:y+self.patch_size, x:x+self.patch_size]
        mask = sample["mask"][:, y:y+self.patch_size, x:x+self.patch_size]
        fov = sample["fov"][:, y:y+self.patch_size, x:x+self.patch_size]

        return {
            "image": image,
            "mask": mask,
            "fov": fov,
            "id": f"{sample['id']}_patch_{idx}",
        }

    def regenerate_patches(self):
        """Regenerate patch locations (call at start of each epoch)."""
        self._generate_patches()


class MergedVesselDataset(Dataset):
    """
    Combined dataset for training on multiple retinal vessel datasets.

    For DP experiments with AdaDPS:
    - Merge DRIVE + STARE + HRF as private training data (105 images)
    - Use single dataset as public data for preconditioner estimation
    """

    def __init__(
        self,
        datasets: List[BaseVesselDataset],
    ):
        """
        Args:
            datasets: List of datasets to combine
        """
        self.datasets = datasets
        self.combined = ConcatDataset(datasets)

        # Compute cumulative sizes for indexing
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return len(self.combined)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.combined[idx]

    def get_dataset_name(self, idx: int) -> str:
        """Get the name of the dataset for a given index."""
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                return self.datasets[i].dataset_name
        return "unknown"


# Keep old name for backwards compatibility
RetinalVesselDataset = MergedVesselDataset


def create_dataloaders(
    drive_dir: Optional[str] = None,
    stare_dir: Optional[str] = None,
    batch_size: int = 4,
    image_size: int = 512,
    num_workers: int = 4,
    validation_fold: int = 0,
    use_patches: bool = False,
    patch_size: int = 128,
    patches_per_image: int = 10,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        drive_dir: Path to preprocessed DRIVE data
        stare_dir: Path to preprocessed STARE data
        batch_size: Batch size
        image_size: Image size (not used if data is already preprocessed)
        num_workers: Number of dataloader workers
        validation_fold: Which fold to use for validation
        use_patches: Whether to use patch-based training
        patch_size: Patch size if using patches
        patches_per_image: Patches per image if using patches

    Returns:
        train_loader, val_loader
    """
    train_transform = get_training_augmentations(image_size)
    val_transform = get_validation_augmentations()

    train_datasets = []
    val_datasets = []

    if drive_dir:
        # DRIVE training set with k-fold split
        drive_train = DRIVEDataset(
            drive_dir,
            split="training",
            transform=train_transform,
            validation_fold=None,  # Use full training set
        )
        drive_val = DRIVEDataset(
            drive_dir,
            split="training",
            transform=val_transform,
            validation_fold=validation_fold,  # Validation fold
        )
        train_datasets.append(drive_train)
        val_datasets.append(drive_val)

    if stare_dir:
        # STARE with k-fold split
        stare_train = STAREDataset(
            stare_dir,
            transform=train_transform,
            validation_fold=validation_fold,
            train=True,
        )
        stare_val = STAREDataset(
            stare_dir,
            transform=val_transform,
            validation_fold=validation_fold,
            train=False,
        )
        train_datasets.append(stare_train)
        val_datasets.append(stare_val)

    # Combine datasets
    if len(train_datasets) > 1:
        train_dataset = RetinalVesselDataset(train_datasets)
        val_dataset = RetinalVesselDataset(val_datasets)
    else:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]

    # Optionally use patch-based training
    if use_patches:
        train_dataset = PatchDataset(
            train_dataset,
            patch_size=patch_size,
            patches_per_image=patches_per_image,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
